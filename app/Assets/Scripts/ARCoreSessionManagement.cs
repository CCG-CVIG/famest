using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ARCoreSessionManagement : MonoBehaviour
{

    [SerializeField] GameObject arCoreSessionPrefab;
    private GameObject newArCoreSessionPrefab;
    //private GoogleARCore.ARCoreSession arcoreSession;


    private void Start()
    {
        newArCoreSessionPrefab = Instantiate(arCoreSessionPrefab, Vector3.zero, Quaternion.identity);
        //arcoreSession = newArCoreSessionPrefab.GetComponent<GoogleARCore.ARCoreSession>();
        //arcoreSession.enabled = true;
    }



    public void Reset()
    {
        StartCoroutine(CreateANewSession());
    }

    IEnumerator CreateANewSession()
    {
        //Destroy
        //arcoreSession.enabled = false;
        if (newArCoreSessionPrefab != null)
            Destroy(newArCoreSessionPrefab);

        yield return new WaitForSeconds(1);

        //Create a new one
        newArCoreSessionPrefab = Instantiate(arCoreSessionPrefab, Vector3.zero, Quaternion.identity);
        //arcoreSession = newArCoreSessionPrefab.GetComponent<GoogleARCore.ARCoreSession>();
        //arcoreSession.enabled = true;
    }
}