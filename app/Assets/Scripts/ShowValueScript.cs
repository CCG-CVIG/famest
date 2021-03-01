using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class ShowValueScript : MonoBehaviour
{
   

    Text sizeValueText;

    /// <summary>
    /// Unity Start methiod initalizing size value
    /// </summary>
    void Start()
    {
        sizeValueText = GetComponent<Text>();
    }

    /// <summary>
    /// Changes value of text marking the size equivalent to sliders position
    /// </summary>
    /// <param name="value">
    /// Size value receaved from Slider
    /// </param>
    /// <returns>
    /// 
    /// </returns>
    public void textUpdate(float value)
    {
        sizeValueText.text = Mathf.RoundToInt(value).ToString();
    }
}
