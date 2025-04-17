const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const CopyWebpackPlugin = require('copy-webpack-plugin');

module.exports = {
  entry: './bootstrap.js', // Entry point that imports the WASM module
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'bootstrap.js',
  },
  mode: 'development', // Use 'production' for optimized builds
  plugins: [
    // Copies the index.html file to the dist folder
    new CopyWebpackPlugin({
        patterns: [
            { from: 'index.html', to: 'index.html' }
        ]
    }),
    // Note: wasm-pack generates the JS bindings and WASM file in ../lca-rs/pkg
    // We expect the user to run `wasm-pack build ../lca-rs --target web` before running webpack
  ],
  experiments: {
    asyncWebAssembly: true, // Enable support for async WASM loading
  },
  devServer: {
    static: {
      directory: path.join(__dirname, 'dist'),
    },
    compress: true,
    port: 8080, // Port for the dev server
  },
  // Optional: If you encounter issues with WASM loading, you might need specific rules
  // module: {
  //   rules: [
  //     {
  //       test: /\.wasm$/,
  //       type: "webassembly/async",
  //     },
  //   ],
  // },
};
