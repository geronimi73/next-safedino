
export function resizeCanvas(canvasOrig, size) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  canvas.height = size.h;
  canvas.width = size.w;

  ctx.drawImage(
    canvasOrig,
    0,
    0,
    canvasOrig.width,
    canvasOrig.height,
    0,
    0,
    canvas.width,
    canvas.height
  );

  return canvas;
}

// Softmax a 1D array
export function softmax1D(arr) {
  // Find max for numerical stability
  const max = Math.max(...arr);
  
  // Compute exp(x - max)
  const exps = arr.map(x => Math.exp(x - max));
  
  // Compute sum
  const sum = exps.reduce((acc, val) => acc + val, 0);
  
  // Normalize
  const softmaxResult = exps.map(exp => exp / sum);
  
  // Return as array
  return softmaxResult
}

/** 
 * input: onnx Tensor [B, *, W, H] and index idx
 * output: Tensor [B, idx, W, H]
 **/
export function sliceTensor(tensor, idx) {
  const [bs, noMasks, width, height] = tensor.dims;
  const stride = width * height;
  const start = stride * idx,
    end = start + stride;

  return tensor.cpuData.slice(start, end);
}

/**
 * input: Float32Array representing ORT.Tensor of shape [1, 1, width, height]
 * output: HTMLCanvasElement (4 channels, RGBA)
 **/
export function float32ArrayToCanvas(array, width, height) {
  const C = 4; // 4 output channels, RGBA
  const imageData = new Uint8ClampedArray(array.length * C);

  for (let srcIdx = 0; srcIdx < array.length; srcIdx++) {
    const trgIdx = srcIdx * C;
    const maskedPx = array[srcIdx] > 0;
    imageData[trgIdx] = maskedPx > 0 ? 0x32 : 0;
    imageData[trgIdx + 1] = maskedPx > 0 ? 0xcd : 0;
    imageData[trgIdx + 2] = maskedPx > 0 > 0 ? 0x32 : 0;
    // imageData[trgIdx + 3] = maskedPx > 0 ? 150 : 0 // alpha
    imageData[trgIdx + 3] = maskedPx > 0 ? 255 : 0; // alpha
  }

  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  canvas.height = height;
  canvas.width = width;
  ctx.putImageData(new ImageData(imageData, width, height), 0, 0);

  return canvas;
}

/** 
 * input: HTMLCanvasElement (RGB)
 * output: Float32Array for later conversion to ORT.Tensor of shape [1, 3, canvas.width, canvas.height]
 *  
 * inspired by: https://onnxruntime.ai/docs/tutorials/web/classify-images-nextjs-github-template.html
 **/ 
export function canvasToFloat32Array(canvas) {
  const imageData = canvas
    .getContext("2d")
    .getImageData(0, 0, canvas.width, canvas.height).data;
  const shape = [1, 3, canvas.width, canvas.height];

  const [redArray, greenArray, blueArray] = [[], [], []];

  for (let i = 0; i < imageData.length; i += 4) {
    redArray.push(imageData[i]);
    greenArray.push(imageData[i + 1]);
    blueArray.push(imageData[i + 2]);
    // skip data[i + 3] to filter out the alpha channel
  }

  const transposedData = redArray.concat(greenArray).concat(blueArray);

  let i,
    l = transposedData.length;
  const float32Array = new Float32Array(shape[1] * shape[2] * shape[3]);
  for (i = 0; i < l; i++) {
    float32Array[i] = transposedData[i] / 255.0; // convert to float
  }

  return { float32Array, shape };
}

