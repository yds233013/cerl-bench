// Empty on purpose. Without a config here, Vite walks up the directory tree and
// finds an unrelated Tailwind PostCSS config outside the repository, which then
// fails to load. This app uses plain CSS and needs no PostCSS plugins.
export default { plugins: {} }
