import { DINO_NSFWCLASSIFIER } from "./dino3_nsfw_classifier";
import { Tensor } from "onnxruntime-web";

const model = new DINO_NSFWCLASSIFIER();

const stats = {
  device: "unknown",
  downloadModelsTime: [],
  encodeImageTimes: [],
  decodeTimes: [],
};

self.onmessage = async (e) => {
  // console.log("worker received message")

  const { type, data } = e.data;

  if (type === "ping") {
    self.postMessage({ type: "downloadInProgress" });
    const startTime = performance.now();
    await model.downloadModels();
    const durationMs = performance.now() - startTime;
    stats.downloadModelsTime.push(durationMs);

    self.postMessage({ type: "loadingInProgress" });
    const report = await model.createSessions();

    stats.device = report.device;

    self.postMessage({ type: "pong", data: report });
    self.postMessage({ type: "stats", data: stats });
  } else if (type === "encodeImage") {
    const { float32Array, shape } = data;
    const imgTensor = new Tensor("float32", float32Array, shape);

    const startTime = performance.now();
    await model.encodeImage(imgTensor);
    const durationMs = performance.now() - startTime;
    stats.encodeImageTimes.push(durationMs);

    self.postMessage({
      type: "encodeImageDone",
      data: { durationMs: durationMs },
    });
    self.postMessage({ type: "stats", data: stats });
  } else if (type === "decodeMask") {
    const {points, maskArray, maskShape} = data;

    const startTime = performance.now();

    let decodingResults 
    if (maskArray) {
      const maskTensor = new Tensor("float32", maskArray, maskShape);
      decodingResults = await model.decode(points, maskTensor); 
    } else {
      decodingResults = await model.decode(points); 
    }
    // decodingResults = Tensor [B=1, Masks, W, H]

    self.postMessage({ type: "decodeMaskResult", data: decodingResults });
    self.postMessage({ type: "stats", data: stats });
  } else if (type === "stats") {
    self.postMessage({ type: "stats", data: stats });
  } else {
    throw new Error(`Unknown message type: ${type}`);
  }
};
