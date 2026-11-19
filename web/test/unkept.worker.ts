// The render worker in a browser that keeps no files for the page, as Firefox's private windows.
import "../src/render.worker";

navigator.storage.getDirectory = () =>
  Promise.reject(new DOMException("The operation is insecure.", "SecurityError"));
