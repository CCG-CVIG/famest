using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

public class LevelLoader : MonoBehaviour
{
    public GameObject loadingScreen;
    public Slider slider;
    public void LoadLevel(int scheneIndex)
    {
        //StartCoroutine(LoadAsynchronusly(scheneIndex));
        SceneManager.LoadScene(scheneIndex);
    }

    IEnumerator LoadAsynchronusly(int sceneIndex)
    {
        AsyncOperation operation = SceneManager.LoadSceneAsync(sceneIndex);

        loadingScreen.SetActive(true);

        while (!operation.isDone)
        {
            float progress = Mathf.Clamp01(operation.progress / .9f);

            slider.value = progress;
            yield return null;
        }
    }
    /*
    public void turnOffARCore(GoogleARCore.ARCoreSession arcoreSession)
    {
        //Destroy
        if (arcoreSession.isActiveAndEnabled)
        {
            arcoreSession.enabled = false;
            Destroy(arcoreSession);
        }
        

    }
    */
}
