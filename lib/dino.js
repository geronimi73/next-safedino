import { AutoProcessor, RawImage } from "@huggingface/transformers";

// Use AutoProcessor from transformers.js. i'm too lazy to lookup what Dinov3 processing does exactly
const MODEL_ID = "onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX";

// Class for interfacing Dino Worker
export default class Dino {
  worker = null

  initialized = false
  modelReady = false
  modelReadyPromise = null
  modelReadyPromiseResolve = null
  processingPromiseResolve = null

  constructor() {
    this.onWorkerMessage = this.onWorkerMessage.bind(this)
  }

  initWorker() {
    if (!this.worker) {
      this.worker = new Worker(new URL('/public/dinoworker.js', import.meta.url))
      this.worker.onerror = function (error) {
        console.error(error.message)
      };
    }

    if (!this.initialized) {
      this.initialized = true
      this.modelReadyPromise = new Promise((resolve, reject) => {
        this.modelReadyPromiseResolve = resolve;
      });
      this.worker.addEventListener('message', this.onWorkerMessage)
      this.worker.postMessage({ type: 'ping' });    // ping the model
    }
  }

  async waitForModelReady() {
    this.initWorker()

    return this.modelReadyPromise
  } 

  async process(canvas) {
    await this.modelReadyPromise

    // Canvas -> RawImage --Preprocess--> Tensor -> Float32Array -> send to worker (Tensor are not serializable, send as Float32Array)
    const processor = await AutoProcessor.from_pretrained(MODEL_ID)
    const image = await RawImage.fromCanvas(canvas);
    const vision_inputs = await processor(image);
    const tensor = vision_inputs.pixel_values.ort_tensor

    // final inputs to the model:
    const float32Array = tensor.cpuData 
    const shape = tensor.dims

    const tensorData = { float32Array, shape }

    this.worker.postMessage({ 
      type: 'process', 
      data: tensorData 
    });

    return new Promise((resolve, reject) => {
      this.processingPromiseResolve = resolve;
    });      
  }

  onWorkerMessage(e) {
    const { type, data } = e.data;
    // console.log(`message received from worker: ${type}`)

    if (type === 'pong') {
      this.modelReady = true
      this.modelReadyPromiseResolve(data)
   
    } else if (type === 'process_result') {
      this.processingPromiseResolve(data)
    }
  }

  destroy() {
    if (this.initialized) {
      this.worker.removeEventListener('message', this.onWorkerMessage)
    }
    if (this.worker) {
      this.worker.terminate()
    }
  }
}
