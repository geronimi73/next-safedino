/** @type {import('next').NextConfig} */

const path = require('path');

const nextConfig = {
    webpack: (config) => {        
        config.resolve.alias = {
          ...config.resolve.alias,
          "sharp$": false,
          "onnxruntime-node$": false,
          "onnxruntime-web": path.join(__dirname, 'node_modules/@huggingface/transformers/node_modules/onnxruntime-web/dist/ort.all.bundle.min.mjs'),
        }

        return config;
    },
}
module.exports = nextConfig
