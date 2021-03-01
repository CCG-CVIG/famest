//using GoogleARCore.Examples.AugmentedImage;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class ChangeScaleScript : MonoBehaviour
{
    public Slider mainSlider;
    public Text text;
    public bool left = false;
    public static GameObject sneaker;
    public static Vector3 baseScale;

    /// <summary>
    /// 
    /// </summary>
    /// <param name="">
    /// 
    /// </param>
    /// <returns>
    /// 
    /// </returns>
    public void Start()
    {
        text.text = "Left";
        
        //Adds a listener to the main slider and invokes a method when the value changes.
        mainSlider.onValueChanged.AddListener(delegate { ValueChangeCheck(); });
    }

    /// <summary>
    /// Invoked when the value of the slider changes.
    /// Scaling selected object to the desired size.
    /// </summary>
    public void ValueChangeCheck()
    {
        var roundValue = mainSlider.value/35;
        if (sneaker != null)
                sneaker.transform.localScale = Vector3.Scale(baseScale, new Vector3(roundValue,roundValue,roundValue));
        Debug.Log(mainSlider.value);
        Debug.Log(sneaker);
    }

    /// <summary>
    /// Switch object side for right to left and change button text.
    /// </summary>
    
    public void changeButtonText()
    {
        if (left)
        {
            text.text = "Left";
            left = false;
            sneaker.transform.GetChild(0).gameObject.SetActive(false);
            sneaker.transform.GetChild(1).gameObject.SetActive(true);
        }
        else
        {
            text.text = "Right";
            left = true;
            sneaker.transform.GetChild(0).gameObject.SetActive(true);
            sneaker.transform.GetChild(1).gameObject.SetActive(false);
        }
    }
}
