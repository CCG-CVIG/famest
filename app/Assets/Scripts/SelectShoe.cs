//using GoogleARCore.Examples.AugmentedImage;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SelectShoe : MonoBehaviour
{
    public GameObject[] SneakerPrefab;
    private GameObject previous = null;

    void OnEnable()
    {
        onClick(0);
    }
        public void onClick(int index)
    {
        //var visualizer = GetComponent<AugmentedImageVisualizer>();
        //Destroy(visualizer.gameObject);
        if (previous != null)
        {
            GameObject.Destroy(previous);
        }
        transform.localPosition = new Vector3(0.0f, 0.0f, 0.0f);
        previous = Instantiate(SneakerPrefab[index], transform);
    }

}
