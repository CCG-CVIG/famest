using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

using TensorFlowLite;
using UnityEngine.UI;
using System.IO;
using System.Collections.Generic;

public class ARCameraImageTFLite : MonoBehaviour
{
    [SerializeField, FilePopup("*.tflite")] string fileName = "coco_ssd_mobilenet_quant.tflite";
    [SerializeField] Canvas canvas = null;
    [SerializeField] Text framePrefab = null;
    [SerializeField, Range(0f, 1f)] float scoreThreshold = 0.5f;
    [SerializeField] TextAsset labelMap = null;
    [SerializeField]
    [Tooltip("The ARCameraManager which will produce frame events.")]
    ARCameraManager m_CameraManager;
    /*
    [SerializeField]
    GameObject indicator;
    */ 
    [SerializeField]
    ARRaycastManager m_RaycastManager;

    /// <summary>
    /// Get or set the <c>ARCameraManager</c>.
    /// </summary>
    public ARCameraManager cameraManager
    {
        get { return m_CameraManager; }
        set { m_CameraManager = value; }
    }


    public ARRaycastManager raycastManager
    {
        get { return m_RaycastManager; }
        set { m_RaycastManager = value; }
    }


    //WebCamTexture webcamTexture;
    SSD ssd;

    Text[] frames;

    public string[] labels;

    public GameObject m_object;
    Texture2D m_Texture;


    // Start is called before the first frame update
    void Start()
    {
        string path = Path.Combine(Application.streamingAssetsPath, fileName);
        ssd = new SSD(path);
        //InitIndicator();

        // Init frames
        frames = new Text[1];
        var parent = cameraManager.transform;

        for (int i = 0; i < frames.Length; i++)
        {
            frames[i] = Instantiate(framePrefab, new Vector3(0, 0, 1), Quaternion.identity, parent);
            //frames[i] = Instantiate(framePrefab, new Vector3(0, 0, 1), Quaternion.identity);
        }
        
        // Labels
        labels = labelMap.text.Split('\n');
    }
    void OnDestroy()
    {
        ssd?.Dispose();
    }

    void SetFrame(Text frame, SSD.Result result, Vector2 size)
    {

        if (result.score < scoreThreshold)
        {
            frame.gameObject.SetActive(false);
            return;
        }
        else
        {
            frame.transform.rotation = cameraManager.transform.rotation;
            frame.gameObject.SetActive(true);

        }

        frame.text = $"{GetLabelName(result.classID)} : {(int)(result.score * 100)}%";
        var rt = frame.transform as RectTransform;
        rt.anchoredPosition = result.rect.position * size - size * 0.5f;
        rt.sizeDelta = result.rect.size * size;

        
        DrawObject(size * 0.5f + rt.anchoredPosition + rt.sizeDelta * 0.5f + new Vector2(3, 0));
    }

    string GetLabelName(int id)
    {
        if (id < 0 || id >= labels.Length - 1)
        {
            return "?";
        }
        return labels[id + 1];
    }

    private void DrawObject(Vector2 vec)
    {
        //var pos = GetPosition(vec);
        var pose = GetPose(vec);
        //Debug.Log(pos);
        if (!pose.position.Equals(Vector3.zero))
        {
            m_object.SetActive(true);
            //m_object.transform.position = pos;
            //m_object.transform.SetPositionAndRotation(pos, new Quaternion(m_object.transform.rotation.x, cameraManager.transform.rotation.y, m_object.transform.rotation.z, m_object.transform.rotation.w));
            m_object.transform.SetPositionAndRotation(pose.position, pose.rotation);
        }
        else
        {
            //m_object.SetActive(false);
        }

    }

    private Vector3 GetPosition(Vector2 vec)
    {
        Debug.Log(vec);
        var hits = new List<ARRaycastHit>();
        //raycastManager.Raycast(new Vector3(vec.x, vec.y, 0), hits, UnityEngine.XR.ARSubsystems.TrackableType.FeaturePoint);

        //if (hits.Count > 0)
        if (raycastManager.Raycast(vec, hits, UnityEngine.XR.ARSubsystems.TrackableType.Planes))
        {
            var pose = hits[0].pose;

            return pose.position;
        }

        return new Vector3(vec.x, vec.y, 1);
    }

    private Pose GetPose(Vector2 vec)
    {
        var hits = new List<ARRaycastHit>();
        if (raycastManager.Raycast(vec, hits, UnityEngine.XR.ARSubsystems.TrackableType.PlaneWithinPolygon))
        {
            var pose = hits[0].pose;

            return pose;
        }

        return new Pose(new Vector3(vec.x, vec.y, 1), Quaternion.identity);
    }
    /*
    public void InitIndicator()
    {
        m_object = Instantiate(indicator, new Vector3(0, 0, -10), Quaternion.identity);
        //m_object.transform.localScale = new Vector3(1.2f, 1.2f, 1.2f);
        m_object.SetActive(false);
    }
    */

    void OnEnable()
    {
        //cameraManager.cameraFrameReceived += OnCameraFrameReceived;
        cameraManager.frameReceived += OnCameraFrameReceived;
    }

    void OnDisable()
    {
        //cameraManager.cameraFrameReceived -= OnCameraFrameReceived;
        cameraManager.frameReceived -= OnCameraFrameReceived;
    }

    unsafe void OnCameraFrameReceived(ARCameraFrameEventArgs eventArgs)
    {
        XRCameraImage image;
        if (!cameraManager.TryGetLatestImage(out image))
            return;

        var conversionParams = new XRCameraImageConversionParams
        {
            // Get the entire image.
            inputRect = new RectInt(0, 0, image.width, image.height),

            // Downsample by 2.
            outputDimensions = new Vector2Int(image.width / 2, image.height / 2),

            // Choose RGBA format.
            outputFormat = TextureFormat.RGBA32,

            // Flip across the vertical axis (mirror image).
            transformation = CameraImageTransformation.MirrorY
        };

        // See how many bytes you need to store the final image.
        int size = image.GetConvertedDataSize(conversionParams);

        // Allocate a buffer to store the image.
        var buffer = new NativeArray<byte>(size, Allocator.Temp);

        // Extract the image data
        image.Convert(conversionParams, new IntPtr(buffer.GetUnsafePtr()), buffer.Length);

        // The image was converted to RGBA32 format and written into the provided buffer
        // so you can dispose of the XRCameraImage. You must do this or it will leak resources.
        image.Dispose();

        // At this point, you can process the image, pass it to a computer vision algorithm, etc.
        // In this example, you apply it to a texture to visualize it.

        // You've got the data; let's put it into a texture so you can visualize it.
        m_Texture = new Texture2D(
            conversionParams.outputDimensions.x,
            conversionParams.outputDimensions.y,
            conversionParams.outputFormat,
            false);

        m_Texture.LoadRawTextureData(buffer);
        m_Texture.Apply();

        Invoque(m_Texture);
        // Done with your temporary data, so you can dispose it.
        buffer.Dispose();
    }

    void Invoque(Texture2D texture)
    {
        ssd.Invoke(texture);

        var results = ssd.GetResults();

        //var size = cameraView.rectTransform.rect.size;
        var size = canvas.pixelRect.size;

        for (int i = 0; i < frames.Length; i++)
        {
            SSD.Result result = results[i];

            // GetLabelName(result.classID)
            SetFrame(frames[i], result, size);
            Debug.Log("Canvas size: " + size);
            Debug.Log("camera image size: " + texture.texelSize);

        }
        //cameraView.material = ssd.transformMat;
        // cameraView.texture = ssd.inputTex;
    }
}
