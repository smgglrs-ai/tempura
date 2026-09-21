use std::collections::HashMap;

struct RadixNode {
    children: HashMap<u32, usize>,
    page_ids: Vec<usize>,
    ref_count: usize,
    last_used: u64,
}

pub struct RadixTree {
    nodes: Vec<RadixNode>,
    tick: u64,
    hits: u64,
    tokens_saved: u64,
}

impl RadixTree {
    pub fn new() -> Self {
        Self {
            nodes: vec![RadixNode {
                children: HashMap::new(),
                page_ids: Vec::new(),
                ref_count: 0,
                last_used: 0,
            }],
            tick: 0,
            hits: 0,
            tokens_saved: 0,
        }
    }

    pub fn insert(&mut self, token_ids: &[u32], page_ids: &[usize]) {
        self.tick += 1;
        let mut node_idx = 0;
        for (i, &token) in token_ids.iter().enumerate() {
            if let Some(&child_idx) = self.nodes[node_idx].children.get(&token) {
                node_idx = child_idx;
                self.nodes[node_idx].last_used = self.tick;
            } else {
                let new_idx = self.nodes.len();
                let page_id = if i < page_ids.len() {
                    vec![page_ids[i]]
                } else {
                    Vec::new()
                };
                self.nodes.push(RadixNode {
                    children: HashMap::new(),
                    page_ids: page_id,
                    ref_count: 0,
                    last_used: self.tick,
                });
                self.nodes[node_idx].children.insert(token, new_idx);
                node_idx = new_idx;
            }
        }
    }

    pub fn lookup(&mut self, token_ids: &[u32]) -> RadixMatch {
        self.tick += 1;
        let mut node_idx = 0;
        let mut matched_len = 0;
        let mut matched_pages = Vec::new();

        for &token in token_ids {
            if let Some(&child_idx) = self.nodes[node_idx].children.get(&token) {
                node_idx = child_idx;
                self.nodes[node_idx].last_used = self.tick;
                self.nodes[node_idx].ref_count += 1;
                matched_len += 1;
                matched_pages.extend_from_slice(&self.nodes[node_idx].page_ids);
            } else {
                break;
            }
        }

        if matched_len > 0 {
            self.hits += 1;
            self.tokens_saved += matched_len as u64;
        }

        RadixMatch {
            matched_len,
            page_ids: matched_pages,
        }
    }

    pub fn release(&mut self, token_ids: &[u32]) {
        let mut node_idx = 0;
        for &token in token_ids {
            if let Some(&child_idx) = self.nodes[node_idx].children.get(&token) {
                node_idx = child_idx;
                self.nodes[node_idx].ref_count = self.nodes[node_idx].ref_count.saturating_sub(1);
            } else {
                break;
            }
        }
    }

    pub fn evict_lru(&mut self) -> Vec<usize> {
        let mut freed_pages = Vec::new();
        let mut to_remove = Vec::new();

        for (idx, node) in self.nodes.iter().enumerate() {
            if idx == 0 {
                continue;
            }
            if node.ref_count == 0 && node.children.is_empty() {
                to_remove.push(idx);
                freed_pages.extend_from_slice(&node.page_ids);
            }
        }

        if let Some(&oldest) = to_remove
            .iter()
            .min_by_key(|&&idx| self.nodes[idx].last_used)
        {
            let freed = self.nodes[oldest].page_ids.clone();
            for node in self.nodes.iter_mut() {
                node.children.retain(|_, &mut child| child != oldest);
            }
            return freed;
        }

        freed_pages
    }

    pub fn hit_count(&self) -> u64 {
        self.hits
    }

    pub fn tokens_saved(&self) -> u64 {
        self.tokens_saved
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

impl Default for RadixTree {
    fn default() -> Self {
        Self::new()
    }
}

pub struct RadixMatch {
    pub matched_len: usize,
    pub page_ids: Vec<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_lookup() {
        let mut tree = RadixTree::new();
        tree.insert(&[1, 2, 3], &[10, 20, 30]);

        let m = tree.lookup(&[1, 2, 3]);
        assert_eq!(m.matched_len, 3);
        assert_eq!(m.page_ids.len(), 3);
        assert_eq!(tree.hit_count(), 1);
        assert_eq!(tree.tokens_saved(), 3);
    }

    #[test]
    fn test_partial_match() {
        let mut tree = RadixTree::new();
        tree.insert(&[1, 2, 3, 4], &[10, 20, 30, 40]);

        let m = tree.lookup(&[1, 2, 5]);
        assert_eq!(m.matched_len, 2);
    }

    #[test]
    fn test_no_match() {
        let mut tree = RadixTree::new();
        tree.insert(&[1, 2, 3], &[10, 20, 30]);

        let m = tree.lookup(&[5, 6, 7]);
        assert_eq!(m.matched_len, 0);
        assert!(m.page_ids.is_empty());
    }

    #[test]
    fn test_shared_prefix() {
        let mut tree = RadixTree::new();
        tree.insert(&[1, 2, 3], &[10, 20, 30]);
        tree.insert(&[1, 2, 4], &[10, 20, 40]);

        let m1 = tree.lookup(&[1, 2, 3]);
        assert_eq!(m1.matched_len, 3);

        let m2 = tree.lookup(&[1, 2, 4]);
        assert_eq!(m2.matched_len, 3);

        let m3 = tree.lookup(&[1, 2]);
        assert_eq!(m3.matched_len, 2);
    }

    #[test]
    fn test_multi_turn_reuse() {
        let mut tree = RadixTree::new();
        let system_prompt = [100, 101, 102, 103];
        tree.insert(&system_prompt, &[0, 1, 2, 3]);

        let turn1 = [100, 101, 102, 103, 200, 201];
        let m1 = tree.lookup(&turn1);
        assert_eq!(m1.matched_len, 4);

        tree.insert(&turn1, &[0, 1, 2, 3, 4, 5]);
        let turn2 = [100, 101, 102, 103, 200, 201, 300];
        let m2 = tree.lookup(&turn2);
        assert_eq!(m2.matched_len, 6);
    }

    #[test]
    fn test_ref_count_and_release() {
        let mut tree = RadixTree::new();
        tree.insert(&[1, 2, 3], &[10, 20, 30]);

        tree.lookup(&[1, 2, 3]);
        tree.lookup(&[1, 2, 3]);
        tree.release(&[1, 2, 3]);

        assert_eq!(tree.hit_count(), 2);
    }

    #[test]
    fn test_evict_lru() {
        let mut tree = RadixTree::new();
        tree.insert(&[1, 2], &[10, 20]);
        tree.insert(&[3, 4], &[30, 40]);

        tree.lookup(&[3, 4]);
        tree.release(&[3, 4]);

        let freed = tree.evict_lru();
        assert!(!freed.is_empty());
    }
}
