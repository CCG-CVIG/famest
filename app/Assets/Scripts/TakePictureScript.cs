
//using GoogleARCore;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.UI;

public class TakePictureScript : MonoBehaviour
{
    private bool camAvailable;
    private WebCamTexture backCam;
    private Texture defaultBackground;

    public RawImage background;
    public AspectRatioFitter fit;
    public Slider slider;
    public LevelLoader levelLoader;
    private int counter;
    private string path;

    public GameObject button;
    private void Start()
    {
        defaultBackground = background.texture;
        WebCamDevice[] devices = WebCamTexture.devices;

        if (devices.Length == 0)
        {
            Debug.Log("No cameras detected");
            camAvailable = false;
            return;
        }

        for (int i = 0; i < devices.Length; i++)
        {
            if (!devices[i].isFrontFacing)
            {
                backCam = new WebCamTexture(devices[i].name, Screen.width, Screen.height);
            }
        }

        if(backCam == null)
        {
            return;
        }

        backCam.Play();
        background.texture = backCam;

        camAvailable = true;
        counter = 0;
        path = Application.persistentDataPath.ToString() + "/Snapshots";
        if (!Directory.Exists(path))
        {
            Directory.CreateDirectory(path);
        }
    }

    private void Update()
    {
        if (!camAvailable)
            return;

        float ratio = (float)backCam.width / (float)backCam.height;
        fit.aspectRatio = ratio;

        float scaleY = backCam.videoVerticallyMirrored ? -1f: 1f;
        background.rectTransform.localScale = new Vector3(1f, scaleY, 1f);

        int orient = -backCam.videoRotationAngle;
        background.rectTransform.localEulerAngles = new Vector3(0, 0, orient);

        if(counter >= 10)
        {
            //TODO
            //foward photos to server

            //start animation
            if (camAvailable)
            {
                backCam.Stop();
            }
            levelLoader.LoadLevel(3);
        }

    }

    //private void Awake()
    //{
    //    _texture = new Texture2D(width, height, _format, false);
    //}

    public void TakePhoto()
    {
        button.SetActive(false);
        Texture2D photo = new Texture2D(backCam.width, backCam.height);
        photo.SetPixels(backCam.GetPixels());
        photo.Apply();

        //Encode to a PNG
        byte[] bytes = photo.EncodeToPNG();
        string filename = "/photo" + counter + ".png";

        File.WriteAllBytes(path + filename, bytes);
        
        counter++;
        slider.value += 0.1f;

        button.SetActive(true);
    }

}
