import { rmSync, mkdirSync, cpSync } from "node:fs";
rmSync("docs", { recursive: true, force: true });
mkdirSync("docs/src", { recursive: true });
for (const f of ["index.html", "styles.css"]) cpSync(f, `docs/${f}`);
for (const f of ["engine.js", "app.js"])
  cpSync(`web/src/${f}`, `docs/src/${f}`);
console.log("docs built");
