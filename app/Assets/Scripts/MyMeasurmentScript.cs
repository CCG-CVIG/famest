using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Security;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading;
using RestSharp;
using UnityEngine;
using UnityEngine.UI;


public class MyMeasurmentScript : MonoBehaviour
{
    public Text footArcVal;
    public Text widthVal;
    public Text heightVal;

    string[] lines;

    // Start is called before the first frame update
    void Start()
    {
        footArcVal.text = "Foot Arc: -- cm";
        widthVal.text = "Width: -- cm";
        heightVal.text = "Height: -- cm";

        // Remove old foot measurements file from folder
        if (File.Exists(Application.persistentDataPath + "/sheet_dimensions.txt"))
        {
            Debug.Log("Removing old measurements...");
            string[] filePaths = Directory.GetFiles(Application.persistentDataPath, "*.txt");
            foreach (string filePath in filePaths)
            {
                File.Delete(filePath);
            }
        }

        ServicePointManager.SecurityProtocol = SecurityProtocolType.Tls12;

        // Request foot measurements
        MeasureRequest();

    }

    /// <summary>
    /// Sets certificate validation for HTTPS
    /// </summary>
    /// <param name="sender"></param>
    /// <param name="certificate"></param>
    /// <param name="chain"></param>
    /// <param name="sslPolicyErrors"></param>
    /// <returns></returns>
    private static bool MyRemoteCertificateValidationCallback(System.Object sender, X509Certificate certificate, X509Chain chain, SslPolicyErrors sslPolicyErrors)
    {
        bool isOk = true;
        // Check if there are errors in the certificate chain and look at each error to determine the cause.
        if (sslPolicyErrors != SslPolicyErrors.None)
        {
            for (int i = 0; i < chain.ChainStatus.Length; i++)
            {
                if (chain.ChainStatus[i].Status == X509ChainStatusFlags.RevocationStatusUnknown)
                {
                    continue;
                }
                chain.ChainPolicy.RevocationFlag = X509RevocationFlag.EntireChain;
                chain.ChainPolicy.RevocationMode = X509RevocationMode.Online;
                chain.ChainPolicy.UrlRetrievalTimeout = new TimeSpan(0, 1, 0);
                chain.ChainPolicy.VerificationFlags = X509VerificationFlags.AllFlags;
                bool chainIsValid = chain.Build((X509Certificate2)certificate);
                if (!chainIsValid)
                {
                    isOk = false;
                    break;
                }
            }
        }
        return isOk;
    }

    void MeasureRequest()
    {
        // Set server certificate validation
        ServicePointManager.ServerCertificateValidationCallback = MyRemoteCertificateValidationCallback;
        // Create client for REST API and send POST request with video file
        var client = new RestClient("https://cloud1cvig.ccg.pt:19100/api");
        //var client = new RestClient("http://192.168.1.117:19100/api");
        var request = new RestRequest("metrics/", Method.GET);
        //IRestResponse response = client.Get(request);
        client.ExecuteAsync(request, response =>
        {
            // Sleep for a sec
            Thread.Sleep(1000);
            if (response.StatusCode == System.Net.HttpStatusCode.OK)
            {
                Debug.Log("Status Code: " + response.StatusCode);
                // Create client for REST API and send POST request for foot measurements file
                //var client = new RestClient("https://cloud1cvig.ccg.pt:19100/api/metrics");
                //client.Timeout = -1;
                //var request = new RestRequest(Method.POST);
                //Debug.Log("Client has sent POST request!");
                //IRestResponse response = client.Execute(request);
                //Debug.Log("Status Code: " + response.StatusCode);

                // Convert response to string and store it in a file
                string measureFilePath = Application.persistentDataPath + "/sheet_dimensions.txt";

                // Check if file path already exists
                if (!File.Exists(measureFilePath))
                {
                    Debug.Log("Creating file path!");
                    using (FileStream filestream = new FileStream(measureFilePath, FileMode.Create, FileAccess.Write, FileShare.ReadWrite))
                    {
                        byte[] info = new UTF8Encoding(true).GetBytes(response.Content.ToString());
                        filestream.Write(info, 0, info.Length);
                    }
                }

                if (File.Exists(measureFilePath))
                {
                    // Read measure file
                    Thread.Sleep(5000);
                    using (FileStream filestream = new FileStream(measureFilePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
                    {
                        if (filestream.CanRead)
                        {
                            List<string> filelines = new List<string>();
                            using (StreamReader reader = new StreamReader(filestream))
                            {
                                while (!reader.EndOfStream)
                                {
                                    filelines.Add(reader.ReadLine());
                                }
                            }
                            lines = filelines.ToArray();
                            if (lines != null)
                            {
                                foreach (string line in lines)
                                {
                                    Debug.Log("Line: " + line);
                                }
                            }
                        }
                    }
                    Thread.Sleep(1000);

                    // Update measurements
                    heightVal.text = lines[0];
                    widthVal.text = lines[1];
                    footArcVal.text = lines[2];
                }
            }
            else
            {
                Debug.Log("Status Code: " + response.StatusCode);
                Debug.Log("Error Response: " + response.ErrorMessage);
                Debug.Log("Error Exception: " + response.ErrorException);
            }
        });
    }

    // Update is called once per frame
    void Update()
    {
        

    }
}
