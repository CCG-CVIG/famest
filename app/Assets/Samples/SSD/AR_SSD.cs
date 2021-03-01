using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using TensorFlowLite;
using UnityEngine.XR.ARFoundation;
using System;

public class AR_SSD : MonoBehaviour
{
    [SerializeField, FilePopup("*.tflite")] string fileName = "coco_ssd_mobilenet_quant.tflite";
    [SerializeField] RawImage cameraView = null;
    [SerializeField] Text framePrefab = null;
    [SerializeField, Range(0f, 1f)] float scoreThreshold = 0.5f;
    [SerializeField] TextAsset labelMap = null;

    [SerializeField]
    [Tooltip("The ARCameraManager which will produce frame events.")]
    ARCameraManager m_CameraManager;

    [SerializeField]
    GameObject indicator;

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

    GameObject m_object;

    void Start()
    {

        string path = Path.Combine(Application.streamingAssetsPath, fileName);
        ssd = new SSD(path);
        InitIndicator();
        
        // Init frames
        frames = new Text[1];
        var parent = cameraView.transform;

        for (int i = 0; i < frames.Length; i++)
        {
            frames[i] = Instantiate(framePrefab, new Vector3(0, 0, 1), Quaternion.identity, parent);
        }

        //m_object = Instantiate(indicator, new Vector3(10, 10, 1), Quaternion.identity);

        // Labels
        labels = labelMap.text.Split('\n');


    }

    void OnDestroy()
    {
        //webcamTexture?.Stop();
        ssd?.Dispose();
    }

    void Update()
    {
        ssd.Invoke(cameraView.texture);

        var results = ssd.GetResults();

        var size = cameraView.rectTransform.rect.size;
        
        for (int i = 0; i < frames.Length; i++)
        {
            SSD.Result result = results[i];
            
            // GetLabelName(result.classID)
            SetFrame(frames[i], result, size);

        }
        cameraView.material = ssd.transformMat;
        // cameraView.texture = ssd.inputTex;
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
            frame.gameObject.SetActive(true);
            
        }
        //Debug.Log(GetPosition(size * 0.5f));

        frame.text = $"{GetLabelName(result.classID)} : {(int)(result.score * 100)}%";
        var rt = frame.transform as RectTransform;
        rt.anchoredPosition = result.rect.position * size - size * 0.5f;        
        rt.sizeDelta = result.rect.size * size;

        //Debug.Log("anchoredPosition: " + rt.anchoredPosition);
        //var vec = size * 0.5f - rt.sizeDelta * 0.5f;
        //Debug.Log("size * 0.5f - rt.sizeDelta * 0.5f: " + vec);

        //DrawObject(result.rect.position * size + result.rect.size * size * new Vector2(0.5f, -0.5f));
        DrawObject(size * 0.5f + rt.anchoredPosition + rt.sizeDelta);
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
        var pos = GetPosition(vec);
        Debug.Log(pos);
        if (!pos.Equals(Vector3.zero))
        {
            m_object.SetActive(true);
            m_object.transform.position = pos;
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
        if(raycastManager.Raycast(vec, hits, UnityEngine.XR.ARSubsystems.TrackableType.Planes))
        {
            var pose = hits[0].pose;

            return pose.position;
        }

        return new Vector3(vec.x, vec.y, 0.5f);
    }

    public void InitIndicator()
    {
        m_object = Instantiate(indicator, new Vector3(0, 0, -1), Quaternion.identity);
        //m_object.transform.localScale = new Vector3(1.2f, 1.2f, 1.2f);
        m_object.SetActive(false);
    }

}

