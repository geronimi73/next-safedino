import path from "path";

import * as ort from "onnxruntime-web/all";
// ort.env.wasm.numThreads=1
// ort.env.wasm.simd = false;

const MODEL_URL = "/dino_v3_linear.onnx";

export class DINO_NSFWCLASSIFIER {
  bufferModel = null;
  sessionModel = null;
  image_encoded = null;

  constructor() {}

  async downloadModels() {
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
        this.bufferModel = await file.arrayBuffer();

        return
      }
    }

    // step 2: download if not cached
    // console.log("File " + filename + " not in cache, downloading from " + url);
    console.log("File not in cache, downloading from " + url);
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

    this.bufferModel = buffer
  }

  async createSessions() {
    /** Creating a session with executionProviders: {"webgpu", "cpu"} fails
     *  => "Error: multiple calls to 'initWasm()' detected."
     *  but ONLY in Safari and Firefox (wtf)
     *  seems to be related to web worker, see https://github.com/microsoft/onnxruntime/issues/22113
     *  => loop through each ep, catch e if not available and move on
     */
    let session = null;
    for (let ep of ["webgpu", "cpu"]) {
      try {
        session = await ort.InferenceSession.create(this.bufferModel, {
          executionProviders: [ep],
        });
      } catch (e) {
        console.error(e);
        continue;
      }

      this.sessionModel = [session, ep]

      return {success: true, device: ep}
    }
    return {success: false}
  }

  async classifyImage(inputTensor) {
    const [session, device] = this.sessionModel
    console.log(session)
    const results = await session.run({ pixel_values: inputTensor });

    console.log(results)

    return results
  }

}
