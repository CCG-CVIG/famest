using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Net;
using System.Net.Sockets;
using System.Net.Http;
using System.Net.Security;
using System.Security.Cryptography.X509Certificates;
using System.Threading;
using RestSharp;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Networking;


public class VideoRecorder : MonoBehaviour
{
    #region private members
    private TcpClient socketConnection;
    private Thread clientReceiveThread;
    private string separator = "<SEPARATOR>";
    private string videoFileName;
    private string filePathName;
    private string plyPathName;
    private string DCIM = "/storage/emulated/0/DCIM";
    private string newPath;
    #endregion

    public LevelLoader levelLoader;
    private void Start()
    {
        // Don't attempt to use the camera if it is already open
        if (NativeCamera.IsCameraBusy())
        {
            return;
        }
        else
        {
            // Connect to TCP Server
            //ConnectToTCPServer();

            // Sleep for 3000ms = 3secs
            //Thread.Sleep(3000);

            // Check connection to TCP server
            //if (socketConnection == null)
            //{
            //    Debug.Log("Cannot send message due to socket connection exception!");
            //    return;
            //}

            ServicePointManager.SecurityProtocol = SecurityProtocolType.Tls12;

            // Remove old surface cloud file from folder
            if (File.Exists(Application.persistentDataPath + "/surface_cloud.ply"))
            {
                Debug.Log("Removing old files...");
                string[] filePaths = Directory.GetFiles(Application.persistentDataPath, "*.ply");
                foreach (string filePath in filePaths)
                {
                    File.Delete(filePath);
                }
            }

            // Create video directory
            //newPath = Application.persistentDataPath + "/Snapshots/";
            newPath = GetAndroidExternalFilesDir() + "/Snapshots/";

            // Check if snapshots path already exists
            if (!Directory.Exists(newPath))
            {
                // Create path folder
                Debug.Log("Creating screenshots directory!");
                Directory.CreateDirectory(newPath);
            }

            ServicePointManager.SecurityProtocol = SecurityProtocolType.Tls12;

            // Record video and send it to REST API
            RecordVideo();
        }
    }

    /// <summary>
    /// Setup TCP socket connection
    /// </summary>
    private void ConnectToTCPServer()
    {
        try
        {
            clientReceiveThread = new Thread(new ThreadStart(ListenForData));
            clientReceiveThread.IsBackground = true;
            clientReceiveThread.Start();
        }
        catch (Exception e)
        {
            Debug.Log("On client connection exception " + e);
        }
    }

    /// <summary>
    /// Runs in background of clientReceiveThread;
    /// Listens for incoming data;
    /// </summary>
    private void ListenForData()
    {
        try
        {
            socketConnection = new TcpClient("192.168.1.117", 30000);
            Byte[] bytes = new byte[1024];
            while (true)
            {
                // Get stream object for reading
                using (NetworkStream stream = socketConnection.GetStream())
                {
                    int length;
                    // Read incoming stream into byte array
                    while ((length = stream.Read(bytes, 0, bytes.Length)) != 0)
                    {
                        var incomingData = new byte[length];
                        Array.Copy(bytes, 0, incomingData, 0, length);
                        // Convert byte array to string message
                        string serverMessage = Encoding.ASCII.GetString(incomingData);
                        Debug.Log("Server message received as: " + serverMessage);
                    }
                }
            }
        }
        catch (SocketException socketException)
        {
            Debug.Log("Socket Exception: " + socketException);
        }
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

    /// <summary>
    /// Records video using NativeCamera and sends file via REST API
    /// </summary>
    private void RecordVideo()
    {
        NativeCamera.Permission permission = NativeCamera.RecordVideo((path) =>
        {
            Debug.Log("Video path: " + path);
            // Create video file name
            videoFileName = "FAMEST" + System.DateTime.Now.ToString("yyyy-MM-dd-HH-mm-ss") + ".mp4";
            if (path != null)
            {
                // Record the video and save it in native gallery
                NativeGallery.SaveVideoToGallery(path, newPath, videoFileName);
                // Sleep for 1000ms = 1sec
                Thread.Sleep(1000);
                // Get video file path name
                filePathName = DCIM + "/(invalid)" + newPath + videoFileName;
                //filePathName = DCIM + "/storage/emulated/0/Android/data/com.ccg.famestcore/files/Snapshots/" + "test1.mp4";
                Debug.Log("File Path: " + filePathName);
                // Get video file information
                FileInfo fileInfo = new FileInfo(filePathName);
                string filesize;
                // Get video file size
                if (fileInfo.Exists)
                {
                    long size = fileInfo.Length;
                    filesize = size.ToString();
                }
                else
                {
                    Debug.Log("Could not locate file!");
                    return;
                }

                // Set server certificate validation
                ServicePointManager.ServerCertificateValidationCallback = MyRemoteCertificateValidationCallback;
                // Create client for REST API and send POST request with video file
                var client = new RestClient("https://cloud1cvig.ccg.pt:19100/api/process");
                //var client = new RestClient("http://192.168.1.117:19100/api/process");
                client.Timeout = -1;
                var request = new RestRequest(Method.POST);
                request.AddFile("file_data", filePathName);
                Debug.Log("Client has sent video file!");
                //IRestResponse response = client.Execute(request);
                client.ExecuteAsync(request, response =>
                {
                    plyPathName = Application.persistentDataPath + "/surface_cloud.ply";
                    // Check if path already exists
                    if (!File.Exists(plyPathName))
                    {
                        Debug.Log("Creating .PLY file path!");
                        using (FileStream filestream = new FileStream(plyPathName, FileMode.Create, FileAccess.Write, FileShare.ReadWrite))
                        {
                            byte[] info = new UTF8Encoding(true).GetBytes(response.Content.ToString());
                            filestream.Write(info, 0, info.Length);
                        }

                    }
                    Debug.Log(response.StatusCode);
                    // Debug path
                    Debug.Log(path);
                    levelLoader.LoadLevel(3);
                });
            }
        }, NativeCamera.Quality.Default, 120);
    }

    /// <summary>
    /// Records video using NativeCamera and sends file to TCP server
    /// </summary>
    //private void RecordVideo()
    //{
    //    try
    //    {
    //        // Get stream object for writting
    //        NetworkStream stream = socketConnection.GetStream();
    //        if (stream.CanWrite)
    //        {
    //            NativeCamera.Permission permission = NativeCamera.RecordVideo((path) =>
    //            {
    //                Debug.Log("Video path: " + path);
    //                string newPath = Application.persistentDataPath.ToString() + "/Snapshots";
    //                // Create video file name
    //                videoFileName = "FAMEST" + System.DateTime.Now.ToString("yyyy-MM-dd-HH-mm-ss") + ".mp4";
    //                // Check if path already exists
    //                if (!Directory.Exists(newPath))
    //                {
    //                    // Create path folder
    //                    Directory.CreateDirectory(newPath);
    //                }
    //                if (path != null)
    //                {
    //                    // Record the video and save it in native gallery
    //                    NativeGallery.SaveVideoToGallery(path, newPath, videoFileName);
    //                    // Sleep for 1000ms = 1sec
    //                    Thread.Sleep(1000);
    //                    // Get video file path name
    //                    filePathName = DCIM + newPath + "/" + videoFileName;
    //                    Debug.Log("File Path: " + filePathName);
    //                    // Get video file information
    //                    FileInfo fileInfo = new FileInfo(filePathName);
    //                    string filesize;
    //                    // Get video file size
    //                    if (fileInfo.Exists)
    //                    {
    //                        long size = fileInfo.Length;
    //                        filesize = size.ToString();
    //                    }
    //                    else
    //                    {
    //                        Debug.Log("Could not locate file!");
    //                        return;
    //                    }
    //                    // Generate message with video file information and send it to the server
    //                    string message = filePathName + separator + filesize;
    //                    byte[] messageBytes = Encoding.ASCII.GetBytes(message);
    //                    // Send message with video file information
    //                    stream.Write(messageBytes, 0, messageBytes.Length);
    //                    Debug.Log("Client has sent message with video file information!");
    //                    // Create file stream, read the file and copy its content to the network stream
    //                    using(FileStream filestream = new FileStream(filePathName, FileMode.Open, FileAccess.Read, FileShare.Read))
    //                    {
    //                        filestream.CopyTo(stream);
    //                    }
    //                    Debug.Log("Client has sent a video file!");
    //                    // Sleep for 3000ms = 3sec
    //                    Thread.Sleep(3000);
    //                    // Close socket connection
    //                    if (!stream.DataAvailable)
    //                    {
    //                        Debug.Log("Closing client connection!");
    //                        socketConnection.Close();
    //                        // Debug path
    //                        Debug.Log(path);
    //                        levelLoader.LoadLevel(3);
    //                    }

    //                    // Debug path
    //                    //Debug.Log(path);
    //                    //levelLoader.LoadLevel(3);
    //                }
    //            }, NativeCamera.Quality.Default, 120);
    //        }
    //    }
    //    catch(SocketException socketException)
    //    {
    //        Debug.Log("Socket exception: " + socketException);
    //    }

    //}

    IEnumerator Upload(byte[] data)
    {
        UnityWebRequest www = UnityWebRequest.Put("http://www.my-server.com/upload", data);
        yield return www.SendWebRequest();

        if (www.isNetworkError || www.isHttpError)
        {
            Debug.Log(www.error);
        }
        else
        {
            Debug.Log("Upload complete!");
        }
    }


    private static string GetAndroidExternalFilesDir()
    {
        using (AndroidJavaClass unityPlayer = new AndroidJavaClass("com.unity3d.player.UnityPlayer"))
        {
            using (AndroidJavaObject context = unityPlayer.GetStatic<AndroidJavaObject>("currentActivity"))
            {
                // Get all available external file directories (emulated and sdCards)
                AndroidJavaObject[] externalFilesDirectories = context.Call<AndroidJavaObject[]>("getExternalFilesDirs", (object)null);
                AndroidJavaObject emulated = null;
                AndroidJavaObject sdCard = null;

                for (int i = 0; i < externalFilesDirectories.Length; i++)
                {
                    AndroidJavaObject directory = externalFilesDirectories[i];
                    using (AndroidJavaClass environment = new AndroidJavaClass("android.os.Environment"))
                    {
                        // Check which one is the emulated and which the sdCard.
                        bool isRemovable = environment.CallStatic<bool>("isExternalStorageRemovable", directory);
                        bool isEmulated = environment.CallStatic<bool>("isExternalStorageEmulated", directory);
                        if (isEmulated)
                            emulated = directory;
                        else if (isRemovable && isEmulated == false)
                            sdCard = directory;
                    }
                }
                // Return the sdCard if available
                if (sdCard != null)
                    return sdCard.Call<string>("getAbsolutePath");
                else
                    return emulated.Call<string>("getAbsolutePath");
            }
        }
    }

}
