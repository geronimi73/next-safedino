import { DINO_NSFWCLASSIFIER } from "./dino3_nsfw_classifier";
import { Tensor } from "onnxruntime-web";

const model = new DINO_NSFWCLASSIFIER();

const stats = {
  device: "unknown",
  downloadModelsTime: [],
  processingTimes: [],
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
  } else if (type === "classifyImage") {
    const { float32Array, shape } = data;
    const imgTensor = new Tensor("float32", float32Array, shape);

    const startTime = performance.now();
    const logits = await model.classifyImage(imgTensor);
    const durationMs = performance.now() - startTime;
    stats.processingTimes.push(durationMs);

    self.postMessage({
      type: "classifyImageResults",
      data: { 
        logits: logits,
        durationMs: durationMs
      },
    });
    self.postMessage({ type: "stats", data: stats });

  } else {
    throw new Error(`Unknown message type: ${type}`);
  }
};
