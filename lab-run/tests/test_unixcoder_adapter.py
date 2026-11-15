"""Offline tiny-model checks: implementation tests, not measured project outcomes."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
READY = all(importlib.util.find_spec(x) for x in ('torch','transformers','tokenizers'))

@unittest.skipUnless(READY, 'Requires optional transformer dependencies')
class UniXcoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from tokenizers import ByteLevelBPETokenizer
        from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
        from unixcoder_adapter import UniXcoderSummarizer
        cls.torch = torch
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        directory = Path(cls.temp.name)
        bpe = ByteLevelBPETokenizer()
        bpe.train_from_iterator(['def add(a,b): return a+b', 'add two numbers'], vocab_size=280,
             special_tokens=['<s>','<pad>','</s>','<unk>','<mask>','<encoder-decoder>','<mask0>'])
        bpe.save_model(str(directory))
        cls.tokenizer = RobertaTokenizer(vocab_file=str(directory/'vocab.json'), merges_file=str(directory/'merges.txt'),
                  additional_special_tokens=['<encoder-decoder>','<mask0>'])
        torch.manual_seed(42)
        config = RobertaConfig(vocab_size=len(cls.tokenizer), hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=128,
            is_decoder=True, hidden_dropout_prob=0, attention_probs_dropout_prob=0)
        cls.model = UniXcoderSummarizer(RobertaModel(config), cls.tokenizer)

    def test_prefix_tokens_and_target_layout(self):
        source=self.model.tokenize(['def add(a,b): return a+b'],length=32)
        ids=self.tokenizer.convert_tokens_to_ids(['<s>','<encoder-decoder>','</s>','<mask0>'])
        self.assertEqual(source[0,:4].tolist(),ids)
        target=self.model.tokenize(['add two numbers'],length=16,target=True)
        self.assertEqual(target[0,0].item(),self.model.sos)
        self.assertEqual(target[0][target[0].ne(self.model.pad)][-1].item(),self.tokenizer.sep_token_id)

    def test_real_training_gradients_and_no_future_target_leak(self):
        m=self.model.train(); m.zero_grad()
        source=m.tokenize(['def add(a,b): return a+b'],length=32)
        target=m.tokenize(['add two numbers'],length=16,target=True)
        loss,logits=m(source,target)
        self.assertTrue(self.torch.isfinite(loss).item()); loss.backward()
        self.assertTrue(m.backbone.embeddings.word_embeddings.weight.grad.abs().sum().item()>0)
        changed=target.clone(); changed[0,4]=8
        _,other=m(source,changed)
        self.torch.testing.assert_close(logits[:,:4],other[:,:4])
        self.assertIs(m.lm_head.weight,m.backbone.embeddings.word_embeddings.weight)

    def test_feature_and_checkpoint_roundtrip(self):
        from unixcoder_adapter import UniXcoderSummarizer
        m=self.model.eval(); source=m.tokenize(['def add(a,b): return a+b'],length=32)
        with self.torch.inference_mode(): before=m.features(source)
        self.assertEqual(tuple(before.shape),(1,16))
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/'checkpoint'; m.save(folder)
            loaded=UniXcoderSummarizer.load(folder).eval()
            with self.torch.inference_mode(): after=loaded.features(source)
            self.torch.testing.assert_close(before,after)
            self.assertEqual(m.summarize(source,3),loaded.summarize(source,3))

    def test_generation_rejects_training_mode_and_excess_length(self):
        source=self.model.tokenize(['x'],length=16)
        with self.assertRaises(ValueError): self.model.train().summarize(source,3)
        with self.assertRaises(ValueError): self.model.eval().summarize(source,200)

if __name__=='__main__': unittest.main()
