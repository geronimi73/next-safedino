"use client";

import React, {useState, useEffect, useRef, useCallback,} from "react";
import { cn } from "@/lib/utils";

// UI
import {Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle} from "@/components/ui/card";
import InputDialog from "@/components/ui/inputdialog";
import { Button } from "@/components/ui/button";
import {LoaderCircle, ImageUp, Github, Fan, Shield, AlertTriangle, Upload, Link} from "lucide-react";

// Dino webworker
import Dino from "@/lib/dino";

export default function Home() {
  // state
  const [device, setDevice] = useState(null);
  const [modelReady, setModelReady] = useState(false);
  const [modelProcessing, setModelProcessing] = useState(false);
  const [modelError, setModelError] = useState(null);
  const [imageReady, setImageReady] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false) // dropzone 
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

  // Image dropzone stuff
  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragOver(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    setIsDragOver(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragOver(false)

    const files = e.dataTransfer.files
    if (files.length > 0) {
      const file = files[0]
      if (file.type.startsWith("image/")) {
        setImageReady(false)
        const dataURL = window.URL.createObjectURL(file)
        setImageURL(dataURL)
      }
    }
  }

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
    <div className="min-h-screen bg-gradient-to-br from-background via-card/30 to-background">
      {/* Banner and Status */}
      <div className="flex items-center justify-center min-h-[calc(100vh-80px)] p-4">
        <div className="w-full max-w-4xl space-y-6">
          <Card className="border-2 border-primary/10 shadow-xl bg-card/80 backdrop-blur-sm">
            <CardHeader className="pb-4">
              {/* Banner */}
              <div className="flex justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2 text-sm md:text-lg font-bold text-card-foreground mb-2">
                  <Button onClick={() =>window.open("https://github.com/geronimi73/next-safedino", "_blank")} variant="ghost" size="sm">
                    <Github className="w-4 h-4" />
                  </Button>
                    Client-side NSFW detection with Meta's DINOv3 
                  </CardTitle>
                </div>

                {/* Status */}
                <div className="flex items-center gap-3 px-4 py-2 bg-muted/50 rounded-lg border">
                  {modelError ? (
                    <>
                      <AlertTriangle className="w-5 h-5 text-destructive" />
                      <span className="text-sm font-medium text-destructive">Error: {modelError}</span>
                    </>
                  ) : !modelReady ? (
                    <>
                      <LoaderCircle className="animate-spin w-5 h-5 text-primary" />
                      <span className="text-sm font-medium text-muted-foreground">Loading model...</span>
                    </>
                  ) : modelProcessing ? (
                    <>
                      <LoaderCircle className="animate-spin w-5 h-5 text-secondary" />
                      <span className="text-sm font-medium text-muted-foreground">Processing image...</span>
                    </>
                  ) : modelReady ? (
                    <>
                      <Fan className="w-5 h-5 text-primary animate-[spin_2.5s_linear_infinite] direction-reverse"/>
                      <span className="text-sm font-medium text-primary">Running on {device}</span>
                    </>
                  ) : (
                    <>
                      <LoaderCircle className="animate-spin w-5 h-5 text-muted-foreground" />
                    </>
                  )}
                </div>
              </div>
            </CardHeader>
          </Card>

          <div className="grid md:grid-cols-2 gap-6">

            {/* Image Card */}
            <Card
              className={`border-2 border-dashed transition-all duration-200 shadow-lg bg-card/80 backdrop-blur-sm ${
                isDragOver ? "border-primary bg-primary/5 scale-[1.02]" : "border-primary/20 hover:border-primary/40"
              }`}
              onDragOver={handleDragOver} onDragLeave={handleDragLeave} onDrop={handleDrop}
            >
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-card-foreground">
                  <Upload className="w-5 h-5" />
                  Upload Image
                </CardTitle>
                <CardDescription>Upload an image file or provide a URL for classification</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {isDragOver && (
                  <div className="absolute inset-0 bg-white bg-opacity-80 rounded-lg flex items-center justify-center z-10">
                    <div className="text-center">
                      <Upload className="w-12 h-12 text-primary mx-auto mb-2" />
                      <p className="text-lg font-semibold text-primary">Drop your image here</p>
                    </div>
                  </div>
                )}

                <div className="flex flex-col gap-3">
                  <Button onClick={() => {fileInputEl.current.click()}} disabled={modelBusy}
                    className="w-full h-12 bg-primary hover:bg-primary/90 text-primary-foreground font-medium"
                  >
                    <ImageUp className="w-5 h-5 mr-2" />
                    Choose File
                  </Button>

                  <Button onClick={() => {setInputDialogOpen(true)}}
                    variant="outline" disabled={modelBusy}
                    className="w-full h-12 border-primary/20 hover:bg-primary/5"
                  >
                    <Link className="w-5 h-5 mr-2" />
                    Load from URL
                  </Button>
                </div>

                <div className="mt-6 p-4 bg-muted/30 rounded-lg border-2 border-muted">
                  <div className="flex justify-center">
                    <canvas
                      ref={canvasEl}
                      className="max-w-full max-h-80 w-auto h-auto rounded-lg shadow-md border border-border"
                    />
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Classification Results card */}
            <Card className="shadow-lg bg-card/80 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-card-foreground">
                  <Shield className="w-5 h-5" />
                  Classification Results
                </CardTitle>
                <CardDescription>AI-powered content safety analysis</CardDescription>
              </CardHeader>
              <CardContent>
                <ClassificationResults result={classificationResult} />
              </CardContent>
            </Card>
          </div>
        </div>
      </div>

      <InputDialog
        open={inputDialogOpen}
        setOpen={setInputDialogOpen}
        submitCallback={handleUrl}
        defaultURL={inputDialogDefaultURL}
      />
      <input ref={fileInputEl} hidden="True" accept="image/*" type="file" onInput={handleFileUpload} />
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

