// PostCSS для Vite: Tailwind + автопрефиксер. Расширение .cjs —
// в package.json нет "type": "module", так что default — CommonJS.
module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
};
