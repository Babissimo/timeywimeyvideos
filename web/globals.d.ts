// What app.js uses of the File System Access API, which Chrome and Edge have and TypeScript's
// DOM types leave out.

interface Window {
  showDirectoryPicker?(options?: { id?: string; startIn?: string }):
    Promise<FileSystemDirectoryHandle>;
}

interface DataTransferItem {
  getAsFileSystemHandle?(): Promise<FileSystemHandle | null>;
}
