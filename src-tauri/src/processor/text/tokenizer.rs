/// phobert tokenizer bpe implementation
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use regex::Regex;

#[allow(dead_code)]
pub struct PhobertTokenizer {
    pub encoder:       HashMap<String, usize>,
    pub bpe_ranks:     HashMap<(String, String), usize>,
    pub cache:         HashMap<String, String>,
    pub unk_token:     String,
    pub bos_token_id:  usize,
    pub pad_token_id:  usize,
    pub eos_token_id:  usize,
    pub sep_token_id:  usize,
    pub cls_token_id:  usize,
    re_word:           Regex,
}

impl PhobertTokenizer {
    pub fn new(
        vocab_file:   &str,
        merges_file:  &str,
        bos_token:    &str,
        eos_token:    &str,
        sep_token:    &str,
        cls_token:    &str,
        unk_token:    &str,
        pad_token:    &str,
        _mask_token:  &str,
    ) -> io::Result<Self> {
        let mut encoder = HashMap::new();
        encoder.insert(bos_token.to_string(), 0);
        encoder.insert(pad_token.to_string(), 1);
        encoder.insert(eos_token.to_string(), 2);
        encoder.insert(unk_token.to_string(), 3);
        Self::add_from_file(&mut encoder, vocab_file)?;

        let mut bpe_ranks = HashMap::new();
        let merges_data = std::fs::read_to_string(merges_file)?;
        for (i, line) in merges_data.lines().enumerate() {
            if line.trim().is_empty() { continue; }
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                bpe_ranks.insert((parts[0].to_string(), parts[1].to_string()), i);
            }
        }

        Ok(PhobertTokenizer {
            bos_token_id: *encoder.get(bos_token).unwrap_or(&0),
            pad_token_id: *encoder.get(pad_token).unwrap_or(&1),
            eos_token_id: *encoder.get(eos_token).unwrap_or(&2),
            sep_token_id: *encoder.get(sep_token).unwrap_or(&2),
            cls_token_id: *encoder.get(cls_token).unwrap_or(&0),
            unk_token: unk_token.to_string(),
            encoder,
            bpe_ranks,
            cache: HashMap::new(),
            re_word: Regex::new(r"\S+\n?").unwrap(),
        })
    }

    fn add_from_file(encoder: &mut HashMap<String, usize>, path: &str) -> io::Result<()> {
        let file   = File::open(path)?;
        let reader = BufReader::new(file);
        for line in reader.lines() {
            let line    = line?;
            let trimmed = line.trim_end();
            if let Some(idx) = trimmed.rfind(' ') {
                let word = &trimmed[..idx];
                if !encoder.contains_key(word) {
                    encoder.insert(word.to_string(), encoder.len());
                }
            }
        }
        Ok(())
    }

    fn get_pairs(word: &[String]) -> Vec<(String, String)> {
        let mut pairs = Vec::new();
        if word.len() < 2 { return pairs; }
        for i in 0..word.len() - 1 {
            pairs.push((word[i].clone(), word[i + 1].clone()));
        }
        pairs
    }

    pub fn bpe(&mut self, token: &str) -> String {
        if let Some(cached) = self.cache.get(token) {
            return cached.clone();
        }
        let mut word: Vec<String> = token.chars().map(|c| c.to_string()).collect();
        if let Some(last) = word.last_mut() {
            last.push_str("</w>");
        }
        if word.len() <= 1 { return token.to_string(); }

        loop {
            let pairs  = Self::get_pairs(&word);
            let bigram = pairs.iter().min_by_key(|pair| {
                self.bpe_ranks.get(*pair).unwrap_or(&usize::MAX)
            });
            match bigram {
                Some(pair) if self.bpe_ranks.contains_key(pair) => {
                    let (first, second) = pair;
                    let mut new_word = Vec::new();
                    let mut i = 0;
                    while i < word.len() {
                        if i < word.len() - 1 && &word[i] == first && &word[i + 1] == second {
                            new_word.push(format!("{}{}", first, second));
                            i += 2;
                        } else {
                            new_word.push(word[i].clone());
                            i += 1;
                        }
                    }
                    word = new_word;
                    if word.len() == 1 { break; }
                }
                _ => break,
            }
        }

        let result = word.join("@@ ");
        let result = if result.ends_with("</w>") {
            result[..result.len() - 4].to_string()
        } else {
            result
        };
        self.cache.insert(token.to_string(), result.clone());
        result
    }

    pub fn tokenize(&mut self, text: &str) -> Vec<String> {
        let mut split_tokens = Vec::new();
        let words: Vec<&str> = self.re_word.find_iter(text).map(|m| m.as_str()).collect();
        for token in words {
            let bpe_result = self.bpe(token);
            for sub_token in bpe_result.split_whitespace() {
                split_tokens.push(sub_token.to_string());
            }
        }
        split_tokens
    }

    pub fn convert_tokens_to_ids(&self, tokens: &[String]) -> Vec<usize> {
        let unk_id = *self.encoder.get(&self.unk_token).unwrap_or(&3);
        tokens.iter()
            .map(|tok| *self.encoder.get(tok).unwrap_or(&unk_id))
            .collect()
    }

    pub fn build_inputs_with_special_tokens(
        &self,
        ids_0: Vec<usize>,
        ids_1: Option<Vec<usize>>,
    ) -> Vec<usize> {
        match ids_1 {
            None => {
                let mut res = vec![self.bos_token_id];
                res.extend(ids_0);
                res.push(self.eos_token_id);
                res
            }
            Some(ids_1) => {
                let mut res = vec![self.cls_token_id];
                res.extend(ids_0);
                res.push(self.sep_token_id);
                res.push(self.sep_token_id);
                res.extend(ids_1);
                res.push(self.sep_token_id);
                res.push(self.eos_token_id);
                res
            }
        }
    }
}
