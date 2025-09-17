"use client";

import React, {useState, useEffect, useRef, useCallback,} from "react";
import { cn } from "@/lib/utils";

// UI
import {Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle} from "@/components/ui/card";
import InputDialog from "@/components/ui/inputdialog";
import { Button } from "@/components/ui/button";
import {LoaderCircle, ImageUp, ImageDown, Github, Fan, Shield, AlertTriangle} from "lucide-react";

// Dino webworker
import Dino from "@/lib/dino";

export default function Home() {
  // state
  const [device, setDevice] = useState(null);
  const [modelReady, setModelReady] = useState(false);
  const [modelProcessing, setModelProcessing] = useState(false);
  const [modelError, setModelError] = useState(null);
  const [imageReady, setImageReady] = useState(false);
  const [inputDialogOpen, setInputDialogOpen] = useState(false);
  const modelBusy = !modelReady || modelProcessing || modelError

  const [classificationResult, setClassificationResult] = useState(null)

  // default image on load 
  const [imageURL, setImageURL] = useState(
    "https://upload.wikimedia.org/wikipedia/commons/8/8e/Laura_Chaubard_%26_Yann_Le_Cun_-_2024_%2853814052697%29_%28cropped%29.jpg"
  );
  // default image URL
  const inputDialogDefaultURL = "https://www.cumfaceai.com/anime-large.png"

  const model = useRef(null);
  const canvasEl = useRef(null);
  const fileInputEl = useRef(null);

  async function loadModel() {
    model.current = new Dino()
    const {success, device} = await model.current.waitForModelReady()

    if (success) {
      setDevice(device)
      setModelReady(true)
    } else {
      setModelError("Model loading error, check JS console")
    }
  }

  async function runNSFWCheck() {
    setModelProcessing(true)

    const [logits, probs] = await model.current.process(canvasEl.current)
    setClassificationResult(probs)

    setModelProcessing(false)
  }

  // New image: From File
  const handleFileUpload = (e) => {
    setImageReady(false)

    const file = e.target.files[0]
    const dataURL = window.URL.createObjectURL(file)

    setImageURL(dataURL)
  }

  // New image: From URL 
  const handleUrl = (urlText) => {
    setImageReady(false)

    const dataURL = urlText;

    setImageURL(dataURL);
  };

  // Image and model ready -> process
  useEffect(() => {
    if (modelReady && imageReady) {
      runNSFWCheck()
    }
  }, [modelReady, imageReady]);

  // New image -> draw onto canvas
  useEffect(() => {
    if (imageURL) {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.src = imageURL;

      img.onload = function () {
        const canvas = canvasEl.current
        canvas.width = img.naturalWidth
        canvas.height = img.naturalHeight
        canvas.getContext("2d").drawImage(img, 0, 0, img.naturalWidth, img.naturalHeight);

        setImageReady(true)
      };
    }
  }, [imageURL]);

  // Init dino model
  useEffect(() => {
    if (!model.current) {
      loadModel()
    }
  }, []);

  return (
    <div className="flex items-center justify-center min-h-screen bg-background p-4">
      <Card className="w-full max-w-2xl">
        <div className="absolute top-4 right-4">
          <Button
            variant="outline"
            size="sm"
            onClick={() =>window.open("https://github.com/geronimi73/next-safedino", "_blank")}
          >
            <Github className="w-4 h-4 mr-2" />
            View on GitHub
          </Button>
        </div>
        <CardHeader>
          <CardTitle>
            <div className="flex flex-col gap-2">
              <p>Clientside NSFW detection with onnxruntime-web and Meta's DINOv3</p>
              <p className="flex gap-1 items-center">
                { modelError ? (
                  <>
                    <span className="text-red-500">Error: {modelError}</span> 
                  </>
                  ) : !modelReady ? (
                  <>
                    <LoaderCircle className="animate-spin w-6 h-6" />
                    Loading model
                  </>
                  ) : modelProcessing ? (
                  <>
                    <LoaderCircle className="animate-spin w-6 h-6" />
                    Processing image
                  </>
                  ) : modelReady ? (
                  <>
                    <Fan color="#000" className="w-6 h-6 animate-[spin_2.5s_linear_infinite] direction-reverse"/>
                    Running on {device}
                  </>
                  ) : (
                  <>
                    <LoaderCircle className="animate-spin w-6 h-6" />
                  </>
                  )
                }
              </p>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col gap-4">

            {/* BUTTONS */}
            <div className="flex justify-between gap-2">
              <div/>
              <div className="flex gap-1">
                {/* Image Upload */}
                <Button onClick={()=>{fileInputEl.current.click()}} variant="secondary" disabled={modelBusy}>
                  <ImageUp/> Upload
                </Button>

                {/* Image from URL */}
                <Button onClick={()=>{setInputDialogOpen(true)}} variant="secondary" disabled={modelBusy}>
                  <ImageUp/> From URL
                </Button>
              </div>
            </div>

            {/* NSFW Prob. */}
            <ClassificationResults result={classificationResult} />

            {/* IMAGE */}
            <div className="flex justify-center">
              <canvas ref={canvasEl} className="max-w-sm w-auto h-auto"/>
            </div>
          </div>
        </CardContent>
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
    </div>
  );
}

const ClassificationResults = ({ result }) => {
  if (!result) return null

  const [ probSafe, probNSFW ] = result
  const nsfwPercentage = Math.round(probNSFW * 100)
  const safePercentage = Math.round(probSafe * 100)
  const isSafe = probSafe > probNSFW

  return (
    <div className="mt-4 p-4 border rounded-lg bg-gradient-to-r from-slate-50 to-slate-100">
      <div className="flex items-center gap-2 mb-3">
        {isSafe ? <Shield className="w-5 h-5 text-green-600" /> : <AlertTriangle className="w-5 h-5 text-red-600" />}
        <h3 className="font-semibold text-lg">Classification: {isSafe ? "Safe" : "NSFW"}</h3>
      </div>

      <div className="space-y-3">
        {/* Safe probability bar */}
        <div className="space-y-1">
          <div className="flex justify-between text-sm">
            <span className="text-green-700 font-medium">Safe</span>
            <span className="text-green-700 font-bold">{safePercentage}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-3">
            <div
              className="bg-gradient-to-r from-green-500 to-green-600 h-3 rounded-full transition-all duration-500 ease-out"
              style={{ width: `${safePercentage}%` }}
            ></div>
          </div>
        </div>

        {/* NSFW probability bar */}
        <div className="space-y-1">
          <div className="flex justify-between text-sm">
            <span className="text-red-700 font-medium">NSFW</span>
            <span className="text-red-700 font-bold">{nsfwPercentage}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-3">
            <div
              className="bg-gradient-to-r from-red-500 to-red-600 h-3 rounded-full transition-all duration-500 ease-out"
              style={{ width: `${nsfwPercentage}%` }}
            ></div>
          </div>
        </div>
      </div>
    </div>
  )
}

