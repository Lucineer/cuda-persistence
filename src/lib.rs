/*!
# cuda-persistence

Durable state management for agents.

An agent that loses its state on restart is no better than a goldfish.
Persistence gives agents memory that survives shutdown — checkpoints,
snapshots, and recovery.

- State snapshots (serialize agent state to bytes)
- Checkpointing (periodic state saves)
- Recovery (restore from last checkpoint)
- Versioning (multiple snapshots, rollback)
- Dirty tracking (only save what changed)
*/

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A state snapshot
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Snapshot {
    pub id: String,
    pub agent_id: String,
    pub state_data: Vec<u8>,
    pub timestamp: u64,
    pub version: u64,
    pub checkpoint_type: CheckpointType,
    pub size_bytes: usize,
    pub checksum: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CheckpointType { Full, Incremental, Manual, Emergency }

/// A tracked state field
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrackedField {
    pub name: String,
    pub value: Vec<u8>,
    pub dirty: bool,
    pub last_modified: u64,
}

/// Checkpoint configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckpointConfig {
    pub interval_ms: u64,          // auto-checkpoint interval
    pub max_snapshots: usize,      // keep N most recent
    pub compress: bool,
    pub verify_on_save: bool,
    pub emergency_on_shutdown: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self { CheckpointConfig { interval_ms: 30_000, max_snapshots: 10, compress: false, verify_on_save: true, emergency_on_shutdown: true } }
}

/// Recovery result
#[derive(Clone, Debug)]
pub struct RecoveryResult {
    pub snapshot_id: String,
    pub success: bool,
    pub fields_restored: usize,
    pub version: u64,
    pub age_ms: u64,
}

/// The persistence manager
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PersistenceManager {
    pub agent_id: String,
    pub snapshots: Vec<Snapshot>,
    pub fields: HashMap<String, TrackedField>,
    pub config: CheckpointConfig,
    pub next_version: u64,
    pub last_checkpoint: u64,
    pub total_checkpoints: u32,
    pub total_recoveries: u32,
}

impl PersistenceManager {
    pub fn new(agent_id: &str) -> Self {
        PersistenceManager { agent_id: agent_id.to_string(), snapshots: vec![], fields: HashMap::new(), config: CheckpointConfig::default(), next_version: 1, last_checkpoint: 0, total_checkpoints: 0, total_recoveries: 0 }
    }

    /// Track a state field
    pub fn track(&mut self, name: &str, initial: Vec<u8>) {
        self.fields.insert(name.to_string(), TrackedField { name: name.to_string(), value: initial, dirty: false, last_modified: now() });
    }

    /// Update a tracked field (marks dirty)
    pub fn set(&mut self, name: &str, value: Vec<u8>) {
        if let Some(field) = self.fields.get_mut(name) {
            if field.value != value { field.value = value; field.dirty = true; field.last_modified = now(); }
        }
    }

    /// Get a field value
    pub fn get(&self, name: &str) -> Option<&Vec<u8>> {
        self.fields.get(name).map(|f| &f.value)
    }

    /// Number of dirty fields
    pub fn dirty_count(&self) -> usize { self.fields.values().filter(|f| f.dirty).count() }

    /// Mark all fields clean
    fn mark_clean(&mut self) { for field in self.fields.values_mut() { field.dirty = false; } }

    /// Create a full checkpoint
    pub fn checkpoint(&mut self, cp_type: CheckpointType) -> Option<String> {
        let state_data = self.serialize_state();
        let checksum = simple_checksum(&state_data);
        let id = format!("snap_{}_v{}", now(), self.next_version);

        let snapshot = Snapshot {
            id: id.clone(), agent_id: self.agent_id.clone(), state_data, timestamp: now(),
            version: self.next_version, checkpoint_type: cp_type, size_bytes: 0, checksum,
        };
        self.next_version += 1;
        self.last_checkpoint = now();
        self.total_checkpoints += 1;
        self.mark_clean();

        // Enforce max snapshots
        if self.snapshots.len() >= self.config.max_snapshots { self.snapshots.remove(0); }
        self.snapshots.push(snapshot);
        Some(id)
    }

    /// Auto-checkpoint if interval elapsed and dirty fields exist
    pub fn auto_checkpoint(&mut self) -> Option<String> {
        if self.dirty_count() == 0 { return None; }
        if now() - self.last_checkpoint < self.config.interval_ms { return None; }
        self.checkpoint(CheckpointType::Incremental)
    }

    /// Emergency checkpoint
    pub fn emergency_checkpoint(&mut self) -> Option<String> {
        self.checkpoint(CheckpointType::Emergency)
    }

    /// Recover from latest snapshot
    pub fn recover(&mut self) -> RecoveryResult {
        match self.snapshots.last() {
            Some(snap) => self.recover_from(&snap.id),
            None => RecoveryResult { snapshot_id: String::new(), success: false, fields_restored: 0, version: 0, age_ms: 0 },
        }
    }

    /// Recover from specific snapshot
    pub fn recover_from(&mut self, snapshot_id: &str) -> RecoveryResult {
        let snap = match self.snapshots.iter().find(|s| s.id == snapshot_id) {
            Some(s) => s.clone(),
            None => return RecoveryResult { snapshot_id: snapshot_id.to_string(), success: false, fields_restored: 0, version: 0, age_ms: 0 },
        };

        // Verify checksum
        if self.config.verify_on_save {
            let current_checksum = simple_checksum(&snap.state_data);
            if current_checksum != snap.checksum {
                return RecoveryResult { snapshot_id: snap.id.clone(), success: false, fields_restored: 0, version: snap.version, age_ms: now() - snap.timestamp };
            }
        }

        // Deserialize
        let restored = self.deserialize_state(&snap.state_data);
        self.fields = restored;
        self.total_recoveries += 1;

        RecoveryResult { snapshot_id: snap.id, success: true, fields_restored: self.fields.len(), version: snap.version, age_ms: now() - snap.timestamp }
    }

    /// Rollback to specific version
    pub fn rollback_to_version(&mut self, version: u64) -> RecoveryResult {
        let snap = match self.snapshots.iter().find(|s| s.version == version) {
            Some(s) => s.id.clone(),
            None => return RecoveryResult { snapshot_id: String::new(), success: false, fields_restored: 0, version: 0, age_ms: 0 },
        };
        self.recover_from(&snap)
    }

    /// Serialize all tracked fields
    fn serialize_state(&self) -> Vec<u8> {
        let mut data = Vec::new();
        for (name, field) in &self.fields {
            let name_bytes = name.as_bytes();
            data.extend_from_slice(&(name_bytes.len() as u64).to_le_bytes());
            data.extend_from_slice(name_bytes);
            data.extend_from_slice(&(field.value.len() as u64).to_le_bytes());
            data.extend_from_slice(&field.value);
        }
        data
    }

    /// Deserialize fields from bytes
    fn deserialize_state(&self, data: &[u8]) -> HashMap<String, TrackedField> {
        let mut fields = HashMap::new();
        let mut pos = 0;
        let now_ts = now();
        while pos < data.len() {
            if pos + 8 > data.len() { break; }
            let name_len = u64::from_le_bytes(data[pos..pos+8].try_into().unwrap_or([0;8])) as usize;
            pos += 8;
            if pos + name_len > data.len() { break; }
            let name = String::from_utf8_lossy(&data[pos..pos+name_len]).to_string();
            pos += name_len;
            if pos + 8 > data.len() { break; }
            let val_len = u64::from_le_bytes(data[pos..pos+8].try_into().unwrap_or([0;8])) as usize;
            pos += 8;
            if pos + val_len > data.len() { break; }
            let value = data[pos..pos+val_len].to_vec();
            pos += val_len;
            fields.insert(name, TrackedField { name: name.clone(), value, dirty: false, last_modified: now_ts });
        }
        fields
    }

    /// Snapshot history summary
    pub fn history(&self) -> String {
        let types: String = self.snapshots.iter().map(|s| match s.checkpoint_type { CheckpointType::Full => "F", CheckpointType::Incremental => "I", CheckpointType::Manual => "M", CheckpointType::Emergency => "E" }).collect();
        format!("Persistence: {} snapshots [{}], {} tracked fields, checkpoints={}, recoveries={}", self.snapshots.len(), types, self.fields.len(), self.total_checkpoints, self.total_recoveries)
    }
}

fn simple_checksum(data: &[u8]) -> u64 {
    let mut sum: u64 = 0;
    for (i, &byte) in data.iter().enumerate() { sum = sum.wrapping_add((byte as u64).wrapping_mul(i as u64 + 1)); }
    sum
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_track_and_get() {
        let mut pm = PersistenceManager::new("agent1");
        pm.track("position", b"(1.0, 2.0)".to_vec());
        assert_eq!(pm.get("position"), Some(&b"(1.0, 2.0)".to_vec()));
    }

    #[test]
    fn test_dirty_tracking() {
        let mut pm = PersistenceManager::new("agent1");
        pm.track("x", b"1".to_vec());
        assert_eq!(pm.dirty_count(), 0);
        pm.set("x", b"2".to_vec());
        assert_eq!(pm.dirty_count(), 1);
    }

    #[test]
    fn test_checkpoint() {
        let mut pm = PersistenceManager::new("agent1");
        pm.track("pos", b"(0,0)".to_vec());
        let id = pm.checkpoint(CheckpointType::Full);
        assert!(id.is_some());
        assert_eq!(pm.snapshots.len(), 1);
        assert_eq!(pm.dirty_count(), 0); // marked clean
    }

    #[test]
    fn test_recover() {
        let mut pm = PersistenceManager::new("agent1");
        pm.track("pos", b"(0,0)".to_vec());
        pm.checkpoint(CheckpointType::Full);
        pm.set("pos", b"(5,5)".to_vec());
        let result = pm.recover();
        assert!(result.success);
        assert_eq!(pm.get("pos"), Some(&b"(0,0)".to_vec()));
    }

    #[test]
    fn test_max_snapshots() {
        let mut pm = PersistenceManager::new("agent1");
        pm.config.max_snapshots = 3;
        pm.track("x", b"0".to_vec());
        for _ in 0..5 { pm.set("x", format!("{}", pm.snapshots.len()).into_bytes()); pm.checkpoint(CheckpointType::Full); }
        assert_eq!(pm.snapshots.len(), 3); // oldest evicted
    }

    #[test]
    fn test_checksum_verify() {
        let mut pm = PersistenceManager::new("agent1");
        pm.track("x", b"test".to_vec());
        pm.checkpoint(CheckpointType::Full);
        // Corrupt state data
        pm.snapshots[0].state_data[0] ^= 0xFF;
        let result = pm.recover();
        assert!(!result.success); // checksum mismatch
    }

    #[test]
    fn test_rollback() {
        let mut pm = PersistenceManager::new("agent1");
        pm.track("x", b"v1".to_vec());
        pm.checkpoint(CheckpointType::Full);
        pm.set("x", b"v2".to_vec());
        pm.checkpoint(CheckpointType::Full);
        pm.set("x", b"v3".to_vec());
        let result = pm.rollback_to_version(1);
        assert!(result.success);
        assert_eq!(pm.get("x"), Some(&b"v1".to_vec()));
    }

    #[test]
    fn test_auto_checkpoint() {
        let mut pm = PersistenceManager::new("agent1");
        pm.config.interval_ms = 0; // always checkpoint
        pm.track("x", b"0".to_vec());
        pm.set("x", b"1".to_vec());
        let id = pm.auto_checkpoint();
        assert!(id.is_some());
    }

    #[test]
    fn test_emergency_checkpoint() {
        let mut pm = PersistenceManager::new("agent1");
        pm.track("x", b"critical".to_vec());
        let id = pm.emergency_checkpoint();
        assert!(id.is_some());
        assert_eq!(pm.snapshots.last().unwrap().checkpoint_type, CheckpointType::Emergency);
    }

    #[test]
    fn test_serialize_deserialize_roundtrip() {
        let mut pm = PersistenceManager::new("agent1");
        pm.track("a", b"hello".to_vec());
        pm.track("b", b"world".to_vec());
        let data = pm.serialize_state();
        let restored = pm.deserialize_state(&data);
        assert_eq!(restored.len(), 2);
        assert_eq!(restored["a"].value, b"hello".to_vec());
    }
}
