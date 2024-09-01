/*
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;
using TensorFlowLite;


public sealed class Detector : MonoBehaviour
{
    [SerializeField, FilePopup("*.tflite")] string faceModelFile = "coco_ssd_mobilenet_quant.tflite";
    [SerializeField, FilePopup("*.tflite")] string faceMeshModelFile = "coco_ssd_mobilenet_quant.tflite";
    [SerializeField, FilePopup("*.tflite")] string irisModelFile = "coco_ssd_mobilenet_quant.tflite";
    public AudioSource src;
    public AudioClip beep;
    [SerializeField] bool useLandmarkToDetection = true;
    [SerializeField] RawImage viewWebCam = null;
    [SerializeField] RawImage viewFaceDetect = null;
    [SerializeField] RawImage viewEyeLeft = null;
    [SerializeField] RawImage viewEyeRight = null;
    public Text txt;
    [SerializeField] Material faceMaterial = null;
    [SerializeField] string webCamName = "HD USB Camera";

    FaceMesh m_faceMesh;
    FaceDetect m_faceDetect;
    IrisDetect m_irisLeftDetect;
    IrisDetect m_irisRightDetect;

    PrimitiveDraw m_drawPrimitive;

    Vector3[] m_rtCorners = new Vector3[4];
    MeshFilter m_faceMeshFilter;
    Vector3[] m_faceKeypoints;

    FaceDetect.Result m_faceDetectionResult;
    IrisDetect.Result m_irisLeftResult;
    IrisDetect.Result m_irisRightResult;

    WebCamTexture m_webcamTexture;

    Vector2Int m_capSize;

    void Start()
    {

        m_capSize = new Vector2Int(1280, 720);

        InitMediaPipeModelPaths();

        InitWebCam(webCamName);
        
        InitDrawPrimitves();

        InitFaceMeshRenderer();

    }



    void Update()
    {
        // Face Detection
        if (m_faceDetectionResult == null || !useLandmarkToDetection)
        {
            m_faceDetect.Invoke(m_webcamTexture);
            viewWebCam.material = m_faceDetect.transformMat;
            m_faceDetectionResult = m_faceDetect.GetResults().FirstOrDefault();

            if (m_faceDetectionResult == null)
            {
                return;
            }
        }

        // Face Mesh Landmarks Detection
        m_faceMesh.Invoke(m_webcamTexture, m_faceDetectionResult);
        viewFaceDetect.texture = m_faceMesh.inputTex;
        var faceMeshResult = m_faceMesh.GetResult();

        if (faceMeshResult.score < 0.5f)
        {
            m_faceDetectionResult = null;
            return;
        }

        // Eye and Iris Landmarks Detection
        int[] indxLeftEye = { 33, 133 };
        int[] indxRightEye = { 362, 263 };

        m_irisLeftDetect.Invoke(m_webcamTexture, indxLeftEye, m_faceDetectionResult, faceMeshResult, 1);
        m_irisRightDetect.Invoke(m_webcamTexture, indxRightEye, m_faceDetectionResult, faceMeshResult, -1);

        viewEyeLeft.texture = m_irisLeftDetect.inputTex;
        viewEyeRight.texture = m_irisRightDetect.inputTex;

        m_irisLeftResult = m_irisLeftDetect.GetResult(1);
        m_irisRightResult = m_irisRightDetect.GetResult(-1);

        // Determine Gaze Direction
        string gazeDirection = DetermineGazeDirection(faceMeshResult, m_irisLeftResult, m_irisRightResult);
        Debug.Log($"Gaze Direction: {gazeDirection}");

        // Draw Results
        DrawResults(m_faceDetectionResult, faceMeshResult, m_irisLeftResult, m_irisRightResult);

        if (useLandmarkToDetection)
        {
            m_faceDetectionResult = m_faceMesh.LandmarkToDetection(faceMeshResult);
        }
    }


    private string DetermineGazeDirection(FaceMesh.Result faceMeshResult, IrisDetect.Result irisLeftResult, IrisDetect.Result irisRightResult)
    {
        // Get eye and iris landmarks
        Vector3 leftEyeCenter = faceMeshResult.keypoints[33]; // Approximation using a face landmark
        Vector3 rightEyeCenter = faceMeshResult.keypoints[263]; // Approximation using a face landmark
        Vector3 leftIrisCenter = irisLeftResult.irislandmark[0];
        Vector3 rightIrisCenter = irisRightResult.irislandmark[0];

        // Calculate the midpoint of the eyes
        Vector3 eyeMidpoint = (leftEyeCenter + rightEyeCenter) / 2;

        // Calculate distances from midpoint to irises
        float leftIrisDistance = Mathf.Abs(leftIrisCenter.x - eyeMidpoint.x);
        float rightIrisDistance = Mathf.Abs(rightIrisCenter.x - eyeMidpoint.x);

        // Debug information
        Debug.Log($"Left Iris Distance: {leftIrisDistance}, Right Iris Distance: {rightIrisDistance}, Eye Midpoint: {eyeMidpoint}");

        // Determine the direction based on the distances
        float threshold = 0.01f; // Adjust this threshold as necessary
        float Leftiris = leftIrisDistance*1.15f;
        float Rightiris = rightIrisDistance * 1.1f;

        if (Mathf.Abs(Leftiris - Rightiris) < threshold)
        {
            txt.text = "Center";
            return "Looking Center";
        }  
        else if ((Leftiris) > Rightiris)
        {
            src.clip = beep;
            src.Play();
            txt.text = "Left";
            return "Looking Left";
        }
        else if (Leftiris < Rightiris)
        {
            src.clip = beep;
            src.Play();
            txt.text = "Right";
            return "Looking Right";
        }else
            return "NA";
    }



    private void DrawResults(FaceDetect.Result faceDetectionResult,
                                FaceMesh.Result faceMesh,
                                IrisDetect.Result irisLeftResult,
                                IrisDetect.Result irisRightResult)
    {
        viewWebCam.rectTransform.GetWorldCorners(m_rtCorners);
        Vector3 min = m_rtCorners[0];
        Vector3 max = m_rtCorners[2];

        float zScale = (max.x - min.x) / 2;

        Color32 faceMeshDetectColor = new Color32(30, 150, 255, 255);
        Color32 eyeDetectColor = new Color32(255, 255, 0, 255);
        Color32 irisDetectColor = new Color32(255, 0, 0, 255);

        // TODO check why 6 landmarks are not showing up properly
        //DrawFaceDetection(faceDetectionResult, min, max, new Color32(80,200,255,255), 0.05f);

        DrawFaceMeshDetection(faceMesh, min, max, faceMeshDetectColor, zScale, 0.035f);

        DrawEyeDetection(min, max, irisLeftResult, irisRightResult, eyeDetectColor, zScale, 0.035f);

        DrawIrisDetection(min, max, irisLeftResult, irisRightResult, irisDetectColor, zScale, 0.035f);

        //TODO uncomment this to show up mesh
        //RenderFaceMesh();

    }

    private void DrawEyeDetection(Vector3 min, Vector3 max,
        IrisDetect.Result irisLeftResult,
        IrisDetect.Result irisRightResult,
        Color32 color, float zScale, float pntSize)
    {

        // int[] eye_idx0 = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

        LerpPointArray(irisLeftResult.eyelandmark, min, max, zScale, pntSize, color, "EyeL");

        LerpPointArray(irisRightResult.eyelandmark, min, max, zScale, pntSize, color, "EyeR");

    }

    private void DrawIrisDetection(Vector3 min, Vector3 max,
        IrisDetect.Result irisLeftResult,
        IrisDetect.Result irisRightResult,
        Color32 color, float zScale, float pntSize)
    {

        LerpPointArray(irisLeftResult.irislandmark, min, max, zScale, pntSize, color, "IrisL");

        LerpPointArray(irisRightResult.irislandmark, min, max, zScale, pntSize, color, "IrisR");

    }

    private void LerpPointArray(Vector3[] keypoints, Vector3 min, Vector3 max, float zScale, float pntSize, Color32 color, string detectionType)
    {
        for (int i = 0; i < keypoints.Length; i++)
        {
            Vector3 p = MathTF.Lerp(min, max, new Vector3(keypoints[i].x, keypoints[i].y, keypoints[i].z));
            p.z = keypoints[i].z * zScale;

            switch (detectionType)
            {
                case ("FACE"):
                    m_faceKeypoints[i] = p;
                    break;
                case ("EyeR"):
                    m_irisRightResult.eyelandmark[i] = p;
                    break;
                case ("EyeL"):
                    m_irisLeftResult.eyelandmark[i] = p;
                    break;
                case ("IrisR"):
                    m_irisRightResult.irislandmark[i] = p;
                    break;
                case ("IrisL"):
                    m_irisLeftResult.irislandmark[i] = p;
                    break;
            }

            m_drawPrimitive.Point(p, pntSize);
        }

        m_drawPrimitive.color = color;

        m_drawPrimitive.Apply();

    }

    private void DrawFaceMeshDetection(FaceMesh.Result face, Vector3 min, Vector3 max, Color color, float zScale, float pntSize)
    {
        LerpPointArray(face.keypoints, min, max, zScale, pntSize, color, "FACE");

    }

    private void DrawFaceDetection(FaceDetect.Result detection, Vector3 min, Vector3 max, Color color, float pntSize)
    {
        // Draw Face Detection
        m_drawPrimitive.color = color;
        UnityEngine.Rect rect = MathTF.Lerp(min, max, detection.rect, true);
        m_drawPrimitive.Rect(rect, 0.03f);

        float zScale = (max.x - min.x) / 2;
        for (int i = 0; i < detection.keypoints.Length; i++)
        {
            Vector3 p = MathTF.Lerp(min, max, new Vector3(detection.keypoints[i].x, 1f - detection.keypoints[i].y, 0));
            //Debug.Log("Detection Pnts: " +  detection.keypoints.Length + "\n");
            m_drawPrimitive.Point(p, pntSize);
        }

        m_drawPrimitive.Apply();

    }

    private void RenderFaceMesh()
    {
        // Update Mesh
        FaceMeshBuilder.UpdateMesh(m_faceMeshFilter.sharedMesh, m_faceKeypoints);
    }

    private void InitFaceMeshRenderer()
    {
        // Create Face Mesh Renderer
        {
            var go = new GameObject("Face");
            go.transform.SetParent(transform);
            var faceRenderer = go.AddComponent<MeshRenderer>();
            faceRenderer.material = faceMaterial;

            m_faceMeshFilter = go.AddComponent<MeshFilter>();
            m_faceMeshFilter.sharedMesh = FaceMeshBuilder.CreateMesh();

            m_faceKeypoints = new Vector3[FaceMesh.KEYPOINT_COUNT];
        }
    }

    void OnDestroy()
    {
        m_webcamTexture?.Stop();
        m_faceDetect?.Dispose();
        m_faceMesh?.Dispose();
        m_irisLeftDetect?.Dispose();
        m_irisRightDetect?.Dispose();

        m_drawPrimitive?.Dispose();
    }

    private void InitWebCam(string camName)
    {
        string cameraName = WebCamUtil.FindName(new WebCamUtil.PreferSpec()
        {
            isFrontFacing = true,
            kind = WebCamKind.WideAngle,
        });

        WebCamDevice[] devices = WebCamTexture.devices;

        m_webcamTexture = new WebCamTexture(cameraName, m_capSize.x, m_capSize.y, 30);

        if (devices.Length > 0)
        {
            for (int i = 0; i < devices.Length; i++)
            {
                Debug.Log("[DEBUG Cam Name] " + devices[i].name);
                if (devices[i].name == camName)
                {
                    m_webcamTexture.deviceName = devices[i].name;
                }
            }
        }


        viewWebCam.texture = m_webcamTexture;

        m_webcamTexture.Play();

        Debug.Log($"Starting camera: {cameraName}");

    }

    private void InitMediaPipeModelPaths()
    {
        string detectionPath = Path.Combine(Application.streamingAssetsPath, faceModelFile);
        m_faceDetect = new FaceDetect(detectionPath);

        string faceMeshPath = Path.Combine(Application.streamingAssetsPath, faceMeshModelFile);
        m_faceMesh = new FaceMesh(faceMeshPath);

        string irisDetectionPath = Path.Combine(Application.streamingAssetsPath, irisModelFile);
        m_irisLeftDetect = new IrisDetect(irisDetectionPath);
        m_irisRightDetect = new IrisDetect(irisDetectionPath);

    }

    private void InitDrawPrimitves()
    {
        m_drawPrimitive = new PrimitiveDraw(Camera.main, gameObject.layer);
    }
}
*/


// The Following Code is for the with an sub routine for the 4 seconds threshold

using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;
using TensorFlowLite;


public sealed class Detector : MonoBehaviour
{
    [SerializeField, FilePopup("*.tflite")] string faceModelFile = "coco_ssd_mobilenet_quant.tflite";
    [SerializeField, FilePopup("*.tflite")] string faceMeshModelFile = "coco_ssd_mobilenet_quant.tflite";
    [SerializeField, FilePopup("*.tflite")] string irisModelFile = "coco_ssd_mobilenet_quant.tflite";
    public AudioSource src;
    public AudioClip beep;
    [SerializeField] bool useLandmarkToDetection = true;
    [SerializeField] RawImage viewWebCam = null;
    [SerializeField] RawImage viewFaceDetect = null;
    [SerializeField] RawImage viewEyeLeft = null;
    [SerializeField] RawImage viewEyeRight = null;
    public Text txt;
    [SerializeField] Material faceMaterial = null;
    [SerializeField] string webCamName = "HD USB Camera";

    FaceMesh m_faceMesh;
    FaceDetect m_faceDetect;
    IrisDetect m_irisLeftDetect;
    IrisDetect m_irisRightDetect;

    PrimitiveDraw m_drawPrimitive;

    Vector3[] m_rtCorners = new Vector3[4];
    MeshFilter m_faceMeshFilter;
    Vector3[] m_faceKeypoints;

    FaceDetect.Result m_faceDetectionResult;
    IrisDetect.Result m_irisLeftResult;
    IrisDetect.Result m_irisRightResult;

    WebCamTexture m_webcamTexture;

    Vector2Int m_capSize;

    float gazeTimer = 0f;
    const float gazeThreshold = 4f; 
    string currentGazeDirection = "Center";

    void Start()
    {

        m_capSize = new Vector2Int(1280, 720);

        InitMediaPipeModelPaths();

        InitWebCam(webCamName);

        InitDrawPrimitves();

        InitFaceMeshRenderer();

    }



    void Update()
    {
        // Face Detection
        if (m_faceDetectionResult == null || !useLandmarkToDetection)
        {
            m_faceDetect.Invoke(m_webcamTexture);
            viewWebCam.material = m_faceDetect.transformMat;
            m_faceDetectionResult = m_faceDetect.GetResults().FirstOrDefault();

            if (m_faceDetectionResult == null)
            {
                return;
            }
        }

        // Face Mesh Landmarks Detection
        m_faceMesh.Invoke(m_webcamTexture, m_faceDetectionResult);
        viewFaceDetect.texture = m_faceMesh.inputTex;
        var faceMeshResult = m_faceMesh.GetResult();

        if (faceMeshResult.score < 0.5f)
        {
            m_faceDetectionResult = null;
            return;
        }

        // Eye and Iris Landmarks Detection
        int[] indxLeftEye = { 33, 133 };
        int[] indxRightEye = { 362, 263 };

        m_irisLeftDetect.Invoke(m_webcamTexture, indxLeftEye, m_faceDetectionResult, faceMeshResult, 1);
        m_irisRightDetect.Invoke(m_webcamTexture, indxRightEye, m_faceDetectionResult, faceMeshResult, -1);

        viewEyeLeft.texture = m_irisLeftDetect.inputTex;
        viewEyeRight.texture = m_irisRightDetect.inputTex;

        m_irisLeftResult = m_irisLeftDetect.GetResult(1);
        m_irisRightResult = m_irisRightDetect.GetResult(-1);

        // Determine Gaze Direction
        string gazeDirection = DetermineGazeDirection(faceMeshResult, m_irisLeftResult, m_irisRightResult);
        Debug.Log($"Gaze Direction: {gazeDirection}");

        // Draw Results
        DrawResults(m_faceDetectionResult, faceMeshResult, m_irisLeftResult, m_irisRightResult);

        if (useLandmarkToDetection)
        {
            m_faceDetectionResult = m_faceMesh.LandmarkToDetection(faceMeshResult);
        }

        // Handle Gaze Timer and Beep
        HandleGazeTimer(gazeDirection);
    }


    private string DetermineGazeDirection(FaceMesh.Result faceMeshResult, IrisDetect.Result irisLeftResult, IrisDetect.Result irisRightResult)
    {
        // Get eye and iris landmarks
        Vector3 leftEyeCenter = faceMeshResult.keypoints[33]; // Approximation using a face landmark
        Vector3 rightEyeCenter = faceMeshResult.keypoints[263]; // Approximation using a face landmark
        Vector3 leftIrisCenter = irisLeftResult.irislandmark[0];
        Vector3 rightIrisCenter = irisRightResult.irislandmark[0];

        // Calculate the midpoint of the eyes
        Vector3 eyeMidpoint = (leftEyeCenter + rightEyeCenter) / 2;

        // Calculate distances from midpoint to irises
        float leftIrisDistance = Mathf.Abs(leftIrisCenter.x - eyeMidpoint.x);
        float rightIrisDistance = Mathf.Abs(rightIrisCenter.x - eyeMidpoint.x);

        // Debug information
        Debug.Log($"Left Iris Distance: {leftIrisDistance}, Right Iris Distance: {rightIrisDistance}, Eye Midpoint: {eyeMidpoint}");

        // Determine the direction based on the distances
        float threshold = 0.01f; // Adjust this threshold as necessary
        float Leftiris = leftIrisDistance * 1.15f;
        float Rightiris = rightIrisDistance * 1.1f;

        if (Mathf.Abs(Leftiris - Rightiris) < threshold)
        {
            txt.text = "Center";
            return "Looking Center";
        }
        else if ((Leftiris) > Rightiris)
        {
            txt.text = "Left";
            return "Looking Left";
        }
        else if (Leftiris < Rightiris)
        {
            txt.text = "Right";
            return "Looking Right";
        }
        else
            return "NA";
    }

    private void HandleGazeTimer(string gazeDirection)
    {
        if (gazeDirection != "Looking Center")
        {
            if (gazeDirection != currentGazeDirection)
            {
                currentGazeDirection = gazeDirection;
                gazeTimer = 0f; // Reset the timer if the direction changes
            }

            gazeTimer += Time.deltaTime;

            if (gazeTimer >= gazeThreshold)
            {
                src.clip = beep;
                src.Play();
                gazeTimer = 0f; // Reset the timer after playing the beep
            }
        }
        else
        {
            currentGazeDirection = "Looking Center";
            gazeTimer = 0f; // Reset the timer if looking at the center
        }
    }

    private void DrawResults(FaceDetect.Result faceDetectionResult,
                                FaceMesh.Result faceMesh,
                                IrisDetect.Result irisLeftResult,
                                IrisDetect.Result irisRightResult)
    {
        viewWebCam.rectTransform.GetWorldCorners(m_rtCorners);
        Vector3 min = m_rtCorners[0];
        Vector3 max = m_rtCorners[2];

        float zScale = (max.x - min.x) / 2;

        Color32 faceMeshDetectColor = new Color32(30, 150, 255, 255);
        Color32 eyeDetectColor = new Color32(255, 255, 0, 255);
        Color32 irisDetectColor = new Color32(255, 0, 0, 255);

        // TODO check why 6 landmarks are not showing up properly
        //DrawFaceDetection(faceDetectionResult, min, max, new Color32(80,200,255,255), 0.05f);

        DrawFaceMeshDetection(faceMesh, min, max, faceMeshDetectColor, zScale, 0.035f);

        DrawEyeDetection(min, max, irisLeftResult, irisRightResult, eyeDetectColor, zScale, 0.035f);

        DrawIrisDetection(min, max, irisLeftResult, irisRightResult, irisDetectColor, zScale, 0.035f);

        //TODO uncomment this to show up mesh
        //RenderFaceMesh();

    }

    private void DrawEyeDetection(Vector3 min, Vector3 max,
        IrisDetect.Result irisLeftResult,
        IrisDetect.Result irisRightResult,
        Color32 color, float zScale, float pntSize)
    {

        // int[] eye_idx0 = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

        LerpPointArray(irisLeftResult.eyelandmark, min, max, zScale, pntSize, color, "EyeL");

        LerpPointArray(irisRightResult.eyelandmark, min, max, zScale, pntSize, color, "EyeR");

    }

    private void DrawIrisDetection(Vector3 min, Vector3 max,
        IrisDetect.Result irisLeftResult,
        IrisDetect.Result irisRightResult,
        Color32 color, float zScale, float pntSize)
    {

        LerpPointArray(irisLeftResult.irislandmark, min, max, zScale, pntSize, color, "IrisL");

        LerpPointArray(irisRightResult.irislandmark, min, max, zScale, pntSize, color, "IrisR");

    }

    private void LerpPointArray(Vector3[] keypoints, Vector3 min, Vector3 max, float zScale, float pntSize, Color32 color, string detectionType)
    {
        for (int i = 0; i < keypoints.Length; i++)
        {
            Vector3 p = MathTF.Lerp(min, max, new Vector3(keypoints[i].x, keypoints[i].y, keypoints[i].z));
            p.z = keypoints[i].z * zScale;

            switch (detectionType)
            {
                case ("FACE"):
                    m_faceKeypoints[i] = p;
                    break;
                case ("EyeR"):
                    m_irisRightResult.eyelandmark[i] = p;
                    break;
                case ("EyeL"):
                    m_irisLeftResult.eyelandmark[i] = p;
                    break;
                case ("IrisR"):
                    m_irisRightResult.irislandmark[i] = p;
                    break;
                case ("IrisL"):
                    m_irisLeftResult.irislandmark[i] = p;
                    break;
            }

            m_drawPrimitive.Point(p, pntSize);
        }

        m_drawPrimitive.color = color;

        m_drawPrimitive.Apply();

    }

    private void DrawFaceMeshDetection(FaceMesh.Result face, Vector3 min, Vector3 max, Color color, float zScale, float pntSize)
    {
        LerpPointArray(face.keypoints, min, max, zScale, pntSize, color, "FACE");

    }

    private void DrawFaceDetection(FaceDetect.Result detection, Vector3 min, Vector3 max, Color color, float pntSize)
    {
        // Draw Face Detection
        m_drawPrimitive.color = color;
        UnityEngine.Rect rect = MathTF.Lerp(min, max, detection.rect, true);
        m_drawPrimitive.Rect(rect, 0.03f);

        float zScale = (max.x - min.x) / 2;
        for (int i = 0; i < detection.keypoints.Length; i++)
        {
            Vector3 p = MathTF.Lerp(min, max, new Vector3(detection.keypoints[i].x, 1f - detection.keypoints[i].y, 0));
            //Debug.Log("Detection Pnts: " +  detection.keypoints.Length + "\n");
            m_drawPrimitive.Point(p, pntSize);
        }

        m_drawPrimitive.Apply();

    }

    private void RenderFaceMesh()
    {
        // Update Mesh
        FaceMeshBuilder.UpdateMesh(m_faceMeshFilter.sharedMesh, m_faceKeypoints);
    }

    private void InitFaceMeshRenderer()
    {
        // Create Face Mesh Renderer
        {
            var go = new GameObject("Face");
            go.transform.SetParent(transform);
            var faceRenderer = go.AddComponent<MeshRenderer>();
            faceRenderer.material = faceMaterial;

            m_faceMeshFilter = go.AddComponent<MeshFilter>();
            m_faceMeshFilter.sharedMesh = FaceMeshBuilder.CreateMesh();

            m_faceKeypoints = new Vector3[FaceMesh.KEYPOINT_COUNT];
        }
    }

    void OnDestroy()
    {
        m_webcamTexture?.Stop();
        m_faceDetect?.Dispose();
        m_faceMesh?.Dispose();
        m_irisLeftDetect?.Dispose();
        m_irisRightDetect?.Dispose();

        m_drawPrimitive?.Dispose();
    }

    private void InitWebCam(string camName)
    {
        string cameraName = WebCamUtil.FindName(new WebCamUtil.PreferSpec()
        {
            isFrontFacing = true,
            kind = WebCamKind.WideAngle,
        });

        WebCamDevice[] devices = WebCamTexture.devices;

        m_webcamTexture = new WebCamTexture(cameraName, m_capSize.x, m_capSize.y, 30);

        if (devices.Length > 0)
        {
            for (int i = 0; i < devices.Length; i++)
            {
                Debug.Log("[DEBUG Cam Name] " + devices[i].name);
                if (devices[i].name == camName)
                {
                    m_webcamTexture.deviceName = devices[i].name;
                }
            }
        }


        viewWebCam.texture = m_webcamTexture;

        m_webcamTexture.Play();

        Debug.Log($"Starting camera: {cameraName}");

    }

    private void InitMediaPipeModelPaths()
    {
        string detectionPath = Path.Combine(Application.streamingAssetsPath, faceModelFile);
        m_faceDetect = new FaceDetect(detectionPath);

        string faceMeshPath = Path.Combine(Application.streamingAssetsPath, faceMeshModelFile);
        m_faceMesh = new FaceMesh(faceMeshPath);

        string irisDetectionPath = Path.Combine(Application.streamingAssetsPath, irisModelFile);
        m_irisLeftDetect = new IrisDetect(irisDetectionPath);
        m_irisRightDetect = new IrisDetect(irisDetectionPath);

    }

    private void InitDrawPrimitves()
    {
        m_drawPrimitive = new PrimitiveDraw(Camera.main, gameObject.layer);
    }
}
