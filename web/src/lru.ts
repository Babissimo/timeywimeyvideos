// A memo for results that depend on a few values the live view repeats, which keeps the most
// recently used `capacity` of them.

export class LruCache<K, V> {
  private readonly entries = new Map<K, V>();
  private readonly capacity: number;

  constructor(capacity: number) {
    this.capacity = capacity;
  }

  get(key: K): V | undefined {
    const value = this.entries.get(key);
    if (value !== undefined) this.set(key, value);
    return value;
  }

  set(key: K, value: V): void {
    this.entries.delete(key);  // most recently used goes last
    this.entries.set(key, value);
    if (this.entries.size > this.capacity) {
      this.entries.delete(this.entries.keys().next().value!);
    }
  }
}
