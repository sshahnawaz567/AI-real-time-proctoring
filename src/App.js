import React, { useRef, useState, useEffect } from 'react';
import Webcam from 'react-webcam';
import * as faceapi from '@vladmandic/face-api';
import { PoseLandmarker, FilesetResolver, DrawingUtils, ObjectDetector } from '@mediapipe/tasks-vision';


function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [capturedFace, setCapturedFace] = useState(null);
  const [isFaceCaptured, setIsFaceCaptured] = useState(false);
  const [warningMessage, setWarningMessage] = useState('');
  const [headMovementStatus, setHeadMovementStatus] = useState('');
  const [detectedObjects, setDetectedObjects] = useState([]);
  const [poseLandmarker, setPoseLandmarker] = useState(null);
  const [objectDetector, setObjectDetector] = useState(null);
  const [faceDetected, setFaceDetected] = useState(false);
  const [multiplePeopleDetected, setMultiplePeopleDetected] = useState(false);
  const [horizontalMovement, setHorizontalMovement] = useState(0);
  const [verticalMovement, setVerticalMovement] = useState(0);
  const [detectedNotAllowedObjects, setDetectedNotAllowedObjects] = useState([]);
  const [faceCaptured, setFaceCaptured] = useState(false); // Track capture status
  const [unauthorizedPerson, setUnauthorizedPerson] = useState(false);

  useEffect(() => {
    loadFaceAPI();
    initializeModels();
  }, []);

  const loadFaceAPI = async () => {
    // Load the models from specific files in the 'face-api' folder
    await faceapi.nets.ssdMobilenetv1.loadFromUri('/models/face-api/ssd_mobilenetv1_model-weights_manifest.json');
    await faceapi.nets.faceLandmark68Net.loadFromUri('/models/face-api/face_landmark_68_model-weights_manifest.json');
    await faceapi.nets.faceRecognitionNet.loadFromUri('/models/face-api/face_recognition_model-weights_manifest.json');

};

  const initializeModels = async () => {
    const vision = await FilesetResolver.forVisionTasks("/models/task_vision/wasm");


    const poseLandmarkerInstance = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "/models/pose_landmarker_full.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numPoses: 4,
    });
    setPoseLandmarker(poseLandmarkerInstance);

    const objectDetectorInstance = await ObjectDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "/models/efficientdet_lite0.tflite",
        delegate: "GPU",
      },
      scoreThreshold: 0.5,
      runningMode: "VIDEO",
    });
    setObjectDetector(objectDetectorInstance);
  };

  const captureFace = async () => {
    if (faceCaptured) return; // Prevent multiple captures
  
    const isCentered = await detectPose();
    if (!isCentered) {
      setHeadMovementStatus('⚠️ Align your face in the center before capturing!');
      return;
    }
  
    if (webcamRef.current) {
      const video = webcamRef.current.video;
      const detections = await faceapi.detectSingleFace(video, new faceapi.SsdMobilenetv1Options())
        .withFaceLandmarks()
        .withFaceDescriptor();
  
      if (detections) {
        setCapturedFace(detections.descriptor);
        setIsFaceCaptured(true);
        setFaceCaptured(true); // Mark face as captured
  
        const landmarks = detections.landmarks.positions.map(pos => ({ x: pos._x, y: pos._y }));
        localStorage.setItem('faceLandmarks', JSON.stringify(landmarks));
  
        setHeadMovementStatus('✅ Face captured successfully!');
      } else {
        setHeadMovementStatus('❌ No face detected. Please try again.');
      }
    }
  };

  const verifyFace = async () => { 
    if (!capturedFace || !webcamRef.current || !webcamRef.current.video) {
        console.error("Webcam or video element is not available.");
        return;
    }

    const video = webcamRef.current.video;

    // 🔹 Ensure FaceAPI is loaded
    if (!faceapi.nets.ssdMobilenetv1.params) {
        console.warn("FaceAPI model not loaded yet. Please wait...");
        return;
    }

    // 🔹 Ensure video is fully loaded before processing
    if (video.readyState !== 4) {
        console.warn("Video is not fully loaded yet.");
        return;
    }

    try {
        // 🔹 Lower the confidence threshold slightly
        const options = new faceapi.SsdMobilenetv1Options({ minConfidence: 0.2 });

        // 🔹 Detect all faces with landmarks and descriptors
        const detections = await faceapi.detectAllFaces(video, options)
            .withFaceLandmarks()
            .withFaceDescriptors();

        console.log("Detections:", detections);

        const videoWidth = video.videoWidth;
        const videoHeight = video.videoHeight;

        let centeredFace = detections[0]; // Default to first face if only one detected
        if (detections.length > 1) {
            centeredFace = detections.reduce((prev, curr) =>
                Math.abs(curr.detection.box.x + curr.detection.box.width / 2 - videoWidth / 2) +
                Math.abs(curr.detection.box.y + curr.detection.box.height / 2 - videoHeight / 2) <
                Math.abs(prev.detection.box.x + prev.detection.box.width / 2 - videoWidth / 2) +
                Math.abs(prev.detection.box.y + prev.detection.box.height / 2 - videoHeight / 2)
                    ? curr : prev
            );
        }

        // 🔹 Ensure a valid face descriptor exists
        if (!centeredFace.descriptor) {
            setFaceDetected(false); // No face detected
            return;
        }

        setFaceDetected(true); // Face detected

        // 🧠 Compute Euclidean Distance for Face Verification
        const distance = faceapi.euclideanDistance(capturedFace, centeredFace.descriptor);
        console.log("Euclidean Distance:", distance);

        setUnauthorizedPerson(distance > 0.6); // Update unauthorized status

    } catch (error) {
        console.error("Face verification error:", error);
    }
};

const detectPose = async () => { 
  if (poseLandmarker && webcamRef.current && webcamRef.current.video.readyState === 4) {
    const video = webcamRef.current.video;
    const videoWidth = video.videoWidth;
    const videoHeight = video.videoHeight;

    webcamRef.current.video.width = videoWidth;
    webcamRef.current.video.height = videoHeight;
    canvasRef.current.width = videoWidth;
    canvasRef.current.height = videoHeight;

    const ctx = canvasRef.current.getContext('2d');
    ctx.clearRect(0, 0, videoWidth, videoHeight);

    // 🎯 Draw the Red Box **Always** (Even Before Detecting the Face)
    const boxWidth = videoWidth * 0.3;
    const boxHeight = videoHeight * 0.3;
    const boxX = (videoWidth - boxWidth) / 2;
    const boxY = (videoHeight - boxHeight) / 2;

    ctx.strokeStyle = 'red';
    ctx.lineWidth = 4;
    ctx.strokeRect(boxX, boxY, boxWidth, boxHeight); // ✅ This will ensure the box is always there

    const result = await poseLandmarker.detectForVideo(video, performance.now());
    const drawingUtils = new DrawingUtils(ctx);
    
    let faceDetected = false;
    let multiplePeopleDetected = false;
    let isFaceCentered = false;
    let horizontalMovement = 0;
    let verticalMovement = 0;

    if (result.landmarks && result.landmarks.length > 0) {
      faceDetected = true;

      if (result.landmarks.length > 1) {
        multiplePeopleDetected = true;
      }

      result.landmarks.forEach((landmark) => {
        const nose = landmark[0]; // Nose tip coordinates

        const centerX = 0.5;
        const centerY = 0.5;
        const tolerance = 0.08;

        isFaceCentered =
          Math.abs(nose.x - centerX) < tolerance &&
          Math.abs(nose.y - centerY) < tolerance;

        let storedReference = localStorage.getItem('referenceNosePosition');
        let referenceNose = storedReference ? JSON.parse(storedReference) : null;

        if (!referenceNose || isFaceCentered) {
          referenceNose = { x: nose.x, y: nose.y };
          localStorage.setItem('referenceNosePosition', JSON.stringify(referenceNose));
        }

        horizontalMovement = Math.abs(nose.x - referenceNose.x);
        verticalMovement = Math.abs(nose.y - referenceNose.y);

        drawingUtils.drawLandmarks(landmark, { radius: 5 });
        drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);
      });
    }

    // ✅ Update state variables
    setFaceDetected(faceDetected);
    setMultiplePeopleDetected(multiplePeopleDetected);
    setHorizontalMovement(horizontalMovement);
    setVerticalMovement(verticalMovement);

    return new Promise((resolve) => {
      setTimeout(() => {
        resolve(isFaceCentered);
      }, 300); // 300ms delay for UI update
    });
  }
};


const detectObjects = async () => {
  if (objectDetector && webcamRef.current?.video.readyState === 4) {
    const video = webcamRef.current.video;
    const detectionResult = await objectDetector.detectForVideo(video, performance.now());

    const notAllowedClasses = ['cell phone', 'tablet', 'book', 'remote', 'backpack', 'mouse', 'tv', 'keyboard', 'laptop'];

    const detectedObjects = detectionResult.detections.filter(detection =>
      notAllowedClasses.includes(detection.categories[0]?.categoryName.toLowerCase())
    );

    setDetectedNotAllowedObjects(detectedObjects.length > 0);
  }
};

// 🔹 Function to determine the highest priority warning
const getHighestPriorityWarning = () => {
  if (multiplePeopleDetected) return '🚨 Critical Warning: More than one person detected!';
  if (unauthorizedPerson) return '🚨 Warning: Unauthorized person detected!';
  if (!faceDetected) return '⚠️ Warning: No face detected!';
  if (detectedNotAllowedObjects) return '🚨 Object Not Allowed! Please remove it.';
  if (horizontalMovement > 0.02 || verticalMovement > 0.02) return '⚠️ Head movement detected!';
  return ''; // No warnings
};

useEffect(() => {
  const interval = setInterval(() => {
    detectPose(); // ✅ Always detect pose (ensures the red box appears from the start)

    if (isFaceCaptured) {
      verifyFace();
      detectObjects();

      // 🔹 Set the highest priority warning
      setWarningMessage(getHighestPriorityWarning());
    }
  }, 300); // Runs every 300ms

  return () => clearInterval(interval); // Cleanup function to stop detection when unmounting
}, [poseLandmarker, objectDetector, isFaceCaptured, faceDetected, multiplePeopleDetected, horizontalMovement, verticalMovement, detectedNotAllowedObjects, unauthorizedPerson]);

  

  return (
    <div className="App" style={{ textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', height: '100vh', justifyContent: 'center' }}>
      
      {/* Webcam & Canvas Wrapper */}
      <div style={{ position: 'relative', width: 450, height: 400 }}>
        <Webcam 
          ref={webcamRef} 
          style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', zIndex: 5 }}
        />
        <canvas 
          ref={canvasRef} 
          style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', zIndex: 6 }} 
        />
      </div>

      {/* Button Below Webcam */}
      <button onClick={captureFace} disabled={isFaceCaptured} style={{ marginTop: 20 }}>
        {isFaceCaptured ? 'Face Captured' : 'Capture Face'}
      </button>

      {/* Warning Messages Below Button */}
      <div style={{ marginTop: 10, fontWeight: 'bold', color: 'red' }}>
        {warningMessage}
      </div>
      <div style={{ marginTop: 5, fontWeight: 'bold', color: 'blue' }}>
        {headMovementStatus}
      </div>

    </div>
  );

};
export default App;

