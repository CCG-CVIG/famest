using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class textButton : MonoBehaviour
{
    public Text text;
    public bool left;
    public static GameObject sneaker;

    public void Start()
    {
        text.text = "Left";
        left = true;
        sneaker.transform.Find("Left").gameObject.SetActive(true);
        sneaker.transform.Find("Right").gameObject.SetActive(false);
    }
    public void changeButtonText()
    {
        if (left)
        {
            text.text = "Right";
            left = false;
            sneaker.transform.Find("Right").gameObject.SetActive(true);
            sneaker.transform.Find("Left").gameObject.SetActive(false);
        }
        else
        {
            text.text = "Left";
            left = true;
            sneaker.transform.Find("Left").gameObject.SetActive(true);
            sneaker.transform.Find("Right").gameObject.SetActive(false);
        }
    }
}
