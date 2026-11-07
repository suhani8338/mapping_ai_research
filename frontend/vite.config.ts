import { defineConfig } from "vite";

const repositoryName = process.env.GITHUB_REPOSITORY?.split("/")[1];
const base = repositoryName
  ? repositoryName.endsWith(".github.io")
    ? "/"
    : `/${repositoryName}/`
  : "./";

export default defineConfig({
  base,
  assetsInclude: ["**/*.wasm"],
  worker: {
    format: "es",
  },
});
