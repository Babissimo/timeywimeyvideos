/** What the browser can't render, as the render worker finds it and the page shows it. */
import { Problem } from "./options";

/**
 * A clip the browser can't render but timeslice.py can: one too large for the GPU or the
 * encoder, or in a container or codec the browser can't read. On the page it carries the
 * timeslice.py command that renders it.
 */
export class Unrenderable extends Problem {
  override name = "Unrenderable";
  readonly command: string | null;

  constructor(message: string, command: string | null = null) {
    super(message);
    this.command = command;
  }
}
