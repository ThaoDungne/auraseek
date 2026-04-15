pub mod embedding;

pub use embedding::AuraModel;
pub use embedding::PhobertTokenizer;

pub struct TextProcessor {
    #[allow(dead_code)]
    pub tokenizer: PhobertTokenizer,
}

impl TextProcessor {
    pub fn new(vocab_path: &str, bpe_path: &str) -> std::io::Result<Self> {
        let tokenizer = PhobertTokenizer::new(
            vocab_path,
            bpe_path,
            "<s>", "</s>", "</s>", "<s>", "<unk>", "<pad>", "<mask>",
        )?;
        Ok(Self { tokenizer })
    }

    #[allow(dead_code)]
    pub fn encode(&mut self, text: &str, max_len: usize) -> (Vec<i64>, Vec<i64>) {
        let tokens = self.tokenizer.tokenize(text);
        let token_ids = self.tokenizer.convert_tokens_to_ids(&tokens);
        let full_ids = self.tokenizer.build_inputs_with_special_tokens(token_ids, None);

        let mut input_ids = vec![self.tokenizer.pad_token_id as i64; max_len];
        let mut attention_mask = vec![0i64; max_len];

        for (i, &id) in full_ids.iter().enumerate().take(max_len) {
            input_ids[i] = id as i64;
            attention_mask[i] = 1;
        }
        (input_ids, attention_mask)
    }
}
