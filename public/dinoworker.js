import path from "path";

import { Tensor } from "@huggingface/transformers";
import * as ort from "onnxruntime-web";
import { softmax1D } from "@/lib/imageutils";

const MODEL_URL =  "https://huggingface.co/g-ronimo/dinov3_nsfw_classifier/resolve/main/dino_v3_linear.onnx";

export class ModelSingleton {
  static buffer;
  static session;
  static sessionDevice;

  static async downloadModel() {
    // step 1: check if cached
    const root = await navigator.storage.getDirectory();
    const filename = path.basename(MODEL_URL);

    let fileHandle = await root
      .getFileHandle(filename)
      .catch((e) => console.error("File does not exist:", filename, e));

    if (fileHandle) {
      console.log("File in cache: " + fileHandle);
      const file = await fileHandle.getFile();

      if (file.size > 0) {
        this.buffer = await file.arrayBuffer();

        return
      }
    }

    // step 2: download if not cached
    // console.log("File " + filename + " not in cache, downloading from " + url);
    console.log("File not in cache, downloading from " + MODEL_URL);
    let buffer = null;
    try {
      buffer = await fetch(MODEL_URL, {
        headers: new Headers({
          Origin: location.origin,
        }),
        mode: "cors",
      }).then((response) => response.arrayBuffer());
    } catch (e) {
      console.error("Download of " + MODEL_URL + " failed: ", e);
      return null;
    }

    // step 3: store
    try {
      const fileHandle = await root.getFileHandle(filename, { create: true });
      const writable = await fileHandle.createWritable();
      await writable.write(buffer);
      await writable.close();

      console.log("Stored " + filename);
    } catch (e) {
      console.error("Storage of " + filename + " failed: ", e);
    }

    this.buffer = buffer
  }

  static async getInstance() {
    if (!this.buffer) {
      await this.downloadModel()
    }

    if (!this.session) {
      // try webgpu first, then cpu
      for (let ep of ["webgpu", "cpu"]) {
        try {
          this.session = await ort.InferenceSession.create(this.buffer, {
            executionProviders: [ep],
          });
          this.sessionDevice = ep

          return {session: this.session, device: this.sessionDevice}

        } 
        catch (e) {
          console.error(e);
          continue;
        }
      }
      console.error("Creating model session failed!")
      return null
    }

    return {session: this.session, device: this.sessionDevice}
  }
}

self.onmessage = async (e) => {
  const modelInstance = await ModelSingleton.getInstance();

  // Something went wrong 
  if (!modelInstance) {
    self.postMessage({
      type: 'pong',
      data: {success: false}
    });      
    return
  }

  const { session, device } = modelInstance
  const { type, data } = e.data;
  // console.log(`worker received message ${type}, data:`)

  if (type === 'ping') {
    self.postMessage({
      type: 'pong',
      data: {success: true, device: device}
    });

  } else if (type === 'process') {
    // reconstruct tensor
    const { float32Array, shape } = data;
    const imgTensor = new Tensor("float32", float32Array, shape);

    // run model
    const results = await session.run({ pixel_values: imgTensor });

    // get logits and softmax() -> probs
    const logits = results.classification.cpuData ? results.classification.cpuData : results.classification.data
    const probs = softmax1D(logits)

    self.postMessage({
      type: 'process_result',
      data: [logits, probs]
    });   

  } else {
    throw new Error(`Unknown message type: ${type}`);
  }
}
