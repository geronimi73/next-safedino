"use client";

import React, {
  useState,
  useEffect,
  useRef,
  createContext,
  useCallback,
} from "react";
import { Analytics } from "@vercel/analytics/next";
import { cn } from "@/lib/utils";

// UI
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import InputDialog from "@/components/ui/inputdialog";
import { Button } from "@/components/ui/button";
import {
  LoaderCircle,
  Crop,
  ImageUp,
  ImageDown,
  Github,
  LoaderPinwheel,
  Fan,
} from "lucide-react";

// Image manipulations
import {
  resizeCanvas,
  canvasToFloat32Array,
  softmax1D,
} from "@/lib/imageutils";

import { Tensor } from "onnxruntime-web";

export default function Home() {
  // state
  const [device, setDevice] = useState(null);
  const [modelIsLoading, setModelIsLoading] = useState(false);
  const [modelIsProcessing, setModelIsProcessing] = useState(false);
  const [modelError, setModelError] = useState(false);
  const [status, setStatus] = useState("");

  // web worker, image 
  const worker = useRef(null);
  const [inputImage, setInputImage] = useState(null); // offscreen canvas
  const [imageURL, setImageURL] = useState(
    "https://upload.wikimedia.org/wikipedia/commons/3/38/Flamingos_Laguna_Colorada.jpg"
  );
  const canvasEl = useRef(null);
  const fileInputEl = useRef(null);
  const pointsRef = useRef([]);

  const [stats, setStats] = useState(null);

  // input dialog for custom URLs
  const [inputDialogOpen, setInputDialogOpen] = useState(false);
  const inputDialogDefaultURL = "https://upload.wikimedia.org/wikipedia/commons/9/96/Pro_Air_Martin_404_N255S.jpg"

  // Decoding finished -> parse result and update mask
  const handleClassifyImageResults = (logitsTensor) => {
    const logitsArray = logitsTensor.cpuData ? logitsTensor.cpuData : logitsTensor.data
    const probs = softmax1D(logitsArray)

    const [probSafe, probNSFW] = probs
    console.log(probNSFW)
  };

  // Start encoding image
  const checkImageClick = async () => {
    setModelIsProcessing(true);

    worker.current.postMessage({
      type: "classifyImage",
      data: canvasToFloat32Array(inputImage),
    });

  };

  // Handle web worker messages
  const onWorkerMessage = (event) => {
    const { type, data } = event.data;

    if (type == "pong") {
      const { success, device } = data;

      if (success) {
        setModelIsLoading(false);
        setDevice(device);
        setStatus("NSFW Check");
      } else {
        setModelError("Error (check JS console)");
      }
    } else if (type == "downloadInProgress" || type == "loadingInProgress") {
      setModelIsLoading(true);
    } else if (type == "classifyImageResults") {
      const {logits, durationMs} = data

      console.log("logits!", logits)
      handleClassifyImageResults(logits)
      setModelIsProcessing(false);
    } else if (type == "stats") {
      setStats(data);
    }
  };

  // Reset all the image-based state: points, mask, offscreen canvases .. 
  const resetState = () => {
    pointsRef.current = [];
    setInputImage(null);
  }

  // New image: From File
  const handleFileUpload = (e) => {
    const file = e.target.files[0]
    const dataURL = window.URL.createObjectURL(file)

    resetState()
    setStatus("Encode image")
    setImageURL(dataURL)
  }

  // New image: From URL 
  const handleUrl = (urlText) => {
    const dataURL = urlText;

    resetState()
    setStatus("Encode image");
    setImageURL(dataURL);
  };

  // Load web worker
  useEffect(() => {
    if (!worker.current) {
      setModelIsLoading(true);

      worker.current = new Worker(new URL("./worker.js", import.meta.url), {
        type: "module",
      });
      worker.current.addEventListener("message", onWorkerMessage);
      worker.current.postMessage({ type: "ping" });

    }
  }, [onWorkerMessage, handleClassifyImageResults]);

  // Load image, pad to square and store in offscreen canvas
  useEffect(() => {
    if (imageURL) {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.src = imageURL;
      img.onload = function () {
        // 1: draw image onto (screen) canvas
        const canvas = canvasEl.current
        canvas.width = img.naturalWidth
        canvas.height = img.naturalHeight
        canvas.getContext("2d").drawImage(
          img, 0, 0, img.naturalWidth, img.naturalHeight,
          // 0, 0, 
        );

        // 1: draw reszied image onto (offscreen) canvas
        const inputImageSize = { w: 224, h: 224 };
        const inputCanvas = resizeCanvas(canvas, inputImageSize)
        inputCanvas.width = inputImageSize.w;
        inputCanvas.height = inputImageSize.h;

        setInputImage(inputCanvas)
      };
    }
  }, [imageURL]);

  return (
    <div className="flex items-center justify-center min-h-screen bg-background p-4">
      <Card className="w-full max-w-2xl">
        <div className="absolute top-4 right-4">
          <Button
            variant="outline"
            size="sm"
            onClick={() =>
              window.open("https://github.com/geronimi73/next-safedino", "_blank")
            }
          >
            <Github className="w-4 h-4 mr-2" />
            View on GitHub
          </Button>
        </div>
        <CardHeader>
          <CardTitle>
            <div className="flex flex-col gap-2">
              <p>
                Clientside NSFW detection with onnxruntime-web and Meta's DINOv3
              </p>
              <p
                className={cn(
                  "flex gap-1 items-center",
                  device ? "visible" : "invisible"
                )}
              >
                <Fan
                  color="#000"
                  className="w-6 h-6 animate-[spin_2.5s_linear_infinite] direction-reverse"
                />
                Running on {device}
              </p>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col gap-4">
            <div className="flex justify-between gap-4">
              { (modelIsLoading || modelIsProcessing ) ? (
                <Button disabled={true}>
                  <LoaderCircle className="animate-spin w-6 h-6" />
                </Button>
              ) : (
                <Button onClick={checkImageClick}>
                  NSFW Check
                </Button>
              )}

              <div className="flex gap-1">
                <Button 
                  onClick={()=>{fileInputEl.current.click()}} 
                  variant="secondary" 
                  disabled={modelIsLoading || modelIsProcessing}>
                  <ImageUp/> Upload
                </Button>
                <Button
                    onClick={()=>{setInputDialogOpen(true)}}
                    variant="secondary"
                    disabled={modelIsLoading || modelIsProcessing}
                  >
                  <ImageUp/> From URL
                </Button>
              </div>
            </div>
            <div className="flex justify-center">
              <canvas ref={canvasEl} className="max-w-lg w-auto h-auto"/>
            </div>
          </div>
        </CardContent>
        <div className="flex flex-col p-4 gap-2">
          <pre className="p-4 border-gray-600 bg-gray-100">
            {stats != null && JSON.stringify(stats, null, 2)}
          </pre>
        </div>
      </Card>
      <InputDialog 
        open={inputDialogOpen} 
        setOpen={setInputDialogOpen} 
        submitCallback={handleUrl}
        defaultURL={inputDialogDefaultURL}
        />
      <input 
        ref={fileInputEl} 
        hidden="True" 
        accept="image/*" 
        type='file' 
        onInput={handleFileUpload} 
        />
      <Analytics />
    </div>
  );
}
