"""UniXcoder summarization adapter; prefix-LM layout follows Microsoft's recipe.

Official reference: microsoft/CodeBERT commit
 d9bc3ab5986715c05f55ad706ad864ea30e4acf3,
 UniXcoder/downstream-tasks/code-summarization/{run,model}.py (MIT).
This implementation adds device-independent greedy decoding and feature access.
"""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

MODEL_ID = 'microsoft/unixcoder-base'
MODEL_REVISION = '5604afdc964f6c53782a6813140ade5216b99006'
PRETRAINED_SHA256 = '6732b60037b086557e4c731c3748c99b4fec735d9ffe3c4c70061fd142e6b9d9'
SOURCE_REVISION = 'd9bc3ab5986715c05f55ad706ad864ea30e4acf3'


class UniXcoderSummarizer(nn.Module):
    def __init__(self, backbone, tokenizer):
        super().__init__()
        if not backbone.config.is_decoder:
            raise ValueError('UniXcoder requires is_decoder=True with explicit prefix attention')
        self.backbone, self.tokenizer = backbone, tokenizer
        self.config = backbone.config
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.lm_head.weight = backbone.embeddings.word_embeddings.weight
        self.pad = tokenizer.pad_token_id
        self.sos = tokenizer.convert_tokens_to_ids('<mask0>')
        mode = tokenizer.convert_tokens_to_ids('<encoder-decoder>')
        if self.sos == tokenizer.unk_token_id or mode == tokenizer.unk_token_id:
            raise ValueError('UniXcoder mode tokens must exist in the tokenizer')

    @classmethod
    def from_pretrained(cls, location=MODEL_ID, *, local_files_only=False):
        revision = MODEL_REVISION if str(location) == MODEL_ID else None
        options = dict(revision=revision, local_files_only=local_files_only)
        config = RobertaConfig.from_pretrained(location, **options)
        config.is_decoder = True
        tokenizer = RobertaTokenizer.from_pretrained(location, **options)
        tokenizer.add_tokens(['<mask0>'], special_tokens=True)
        backbone = RobertaModel.from_pretrained(location, config=config, **options)
        if len(tokenizer) != config.vocab_size:
            raise ValueError('Pinned tokenizer/model vocabulary mismatch')
        return cls(backbone, tokenizer)

    def tokenize(self, texts, *, length, target=False):
        overhead = 2 if target else 5
        if length <= overhead:
            raise ValueError('Token length is too short for required special tokens')
        rows = []
        for text in texts:
            tokens = self.tokenizer.tokenize(text)[:length-overhead]
            if target:
                tokens = ['<mask0>'] + tokens + [self.tokenizer.sep_token]
            else:
                tokens = [self.tokenizer.cls_token, '<encoder-decoder>', self.tokenizer.sep_token,
                          '<mask0>'] + tokens + [self.tokenizer.sep_token]
            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            rows.append(ids + [self.pad] * (length-len(ids)))
        if not rows:
            raise ValueError('At least one input is required')
        return torch.tensor(rows, dtype=torch.long, device=next(self.parameters()).device)

    def _encode(self, source):
        keep = source.ne(self.pad)
        return self.backbone(source, attention_mask=keep[:, :, None] & keep[:, None, :], use_cache=True)

    def forward(self, source, target):
        if source.size(1) + target.size(1) > self.config.max_position_embeddings - 2:
            raise ValueError('Combined source/target exceeds positional capacity')
        encoded = self._encode(source)
        ids = torch.cat((source, target), dim=1)
        s, t = source.size(1), target.size(1)
        causal = torch.arange(s+t, device=ids.device)[None, :] <= torch.arange(s, s+t, device=ids.device)[:, None]
        mask = causal[None, :, :] & ids[:, None, :].ne(self.pad)
        hidden = self.backbone(target, attention_mask=mask, past_key_values=encoded.past_key_values,
                               use_cache=False).last_hidden_state
        logits = self.lm_head(hidden)
        labels = target[:, 1:].contiguous()
        if not labels.ne(self.pad).any():
            raise ValueError('Target has no supervised tokens')
        loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)),
                               labels.reshape(-1), ignore_index=self.pad)
        return loss, logits

    def features(self, source):
        hidden = self._encode(source).last_hidden_state
        mask = source.ne(self.pad).unsqueeze(-1)
        return (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)

    @torch.inference_mode()
    def summarize(self, source, max_new_tokens=128):
        if self.training:
            raise ValueError('Call eval() before generation')
        if max_new_tokens < 1 or source.size(1)+max_new_tokens > self.config.max_position_embeddings-2:
            raise ValueError('Invalid generation length')
        answers = []
        for row in source:
            context = row[row.ne(self.pad)].unsqueeze(0)
            cache = self._encode(context).past_key_values
            token = torch.tensor([[self.sos]], device=row.device)
            output = []
            for _ in range(max_new_tokens):
                length = cache[0][0].size(-2)
                mask = torch.ones((1, 1, length+1), device=row.device, dtype=torch.bool)
                result = self.backbone(token, attention_mask=mask, past_key_values=cache, use_cache=True)
                token = self.lm_head(result.last_hidden_state[:, -1]).argmax(-1, keepdim=True)
                cache = result.past_key_values
                if token.item() == self.tokenizer.sep_token_id:
                    break
                output.append(token.item())
            answers.append(self.tokenizer.decode(output, skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False))
        return answers

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=False)
        self.config.save_pretrained(folder)
        self.tokenizer.save_pretrained(folder)
        torch.save(self.state_dict(), folder / 'adapter_model.bin')

    @classmethod
    def load(cls, folder):
        config = RobertaConfig.from_pretrained(folder, local_files_only=True)
        tokenizer = RobertaTokenizer.from_pretrained(folder, local_files_only=True)
        model = cls(RobertaModel(config), tokenizer)
        model.load_state_dict(torch.load(Path(folder)/'adapter_model.bin', map_location='cpu', weights_only=True))
        return model
