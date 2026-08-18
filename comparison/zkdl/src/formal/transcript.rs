use crate::algebra::{Point, Scalar};
use sha2::{Digest, Sha256};

pub struct Transcript { state: Sha256, counter: u64 }

impl Transcript {
    pub fn new(protocol: &str) -> Self {
        let mut state=Sha256::new();
        state.update(b"zkDL-GAT-Fiat-Shamir-v1\0");
        state.update((protocol.len() as u64).to_be_bytes()); state.update(protocol.as_bytes());
        Self{state,counter:0}
    }
    pub fn label(&mut self, label:&str) { self.bytes(b"label",label.as_bytes()); }
    pub fn bytes(&mut self, tag:&[u8], bytes:&[u8]) {
        self.state.update((tag.len() as u64).to_be_bytes()); self.state.update(tag);
        self.state.update((bytes.len() as u64).to_be_bytes()); self.state.update(bytes);
    }
    pub fn u64(&mut self, tag:&[u8], v:u64) { self.bytes(tag,&v.to_be_bytes()); }
    pub fn scalar(&mut self, tag:&[u8], v:Scalar) { self.bytes(tag,&v.0); }
    pub fn point(&mut self, tag:&[u8], v:Point) { self.bytes(tag,&v.0); }
    pub fn challenge(&mut self, label:&str) -> Scalar {
        let mut h=self.state.clone(); h.update(b"challenge\0"); h.update(self.counter.to_be_bytes()); h.update(label.as_bytes());
        let digest=h.finalize(); self.counter+=1;
        let out=Scalar::hash_to_field(b"zkDL-GAT-hash-to-field-v1",&digest);
        self.scalar(b"challenge",out); out
    }
    pub fn digest(&self)->[u8;32] { self.state.clone().finalize().into() }
}
