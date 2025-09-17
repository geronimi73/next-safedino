/** @type {import('next').NextConfig} */

const path = require('path');

const nextConfig = {
    eslint: {
        ignoreDuringBuilds: true,
    },
    typescript: {
        ignoreBuildErrors: true,
    },
    images: {
        unoptimized: true,
    },
    outputFileTracingExcludes: {
        '/': [
            './node_modules/onnxruntime-node/*'
        ],
    },
    webpack: (config) => {        
        config.resolve.alias = {
          ...config.resolve.alias,
          "onnxruntime-web": path.join(__dirname, 'node_modules/@huggingface/transformers/node_modules/onnxruntime-web/dist/ort.all.bundle.min.mjs'),
        }

        return config;
    },
}
module.exports = nextConfig
