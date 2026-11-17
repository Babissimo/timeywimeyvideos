// The render worker where the browser's storage for the page fills up 2 kB into a render.
import "../src/render.worker";

const open = FileSystemFileHandle.prototype.createSyncAccessHandle;
FileSystemFileHandle.prototype.createSyncAccessHandle = async function () {
  const file = await open.call(this);
  const write = file.write.bind(file);
  file.write = (data, options) => {
    if (options.at + data.byteLength <= 2000) return write(data, options);
    throw new DOMException("The quota has been exceeded.", "QuotaExceededError");
  };
  return file;
};
