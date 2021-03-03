using System;
using System.IO;
using System.Text;
using System.Net;
using System.Net.Sockets;
using System.Net.Security;
using System.Threading;
using System.Security.Cryptography.X509Certificates;
using RestSharp;
using ICSharpCode.SharpZipLib.Zip;
using UnityEngine;
using System.Collections;

public class PictureRecorder : MonoBehaviour
{
    #region private members
    private bool takePicture = true;
    private bool compressPictures = false;
    private TcpClient socketConnection;
    private Thread clientReceiveThread;
    private string separator = "<SEPARATOR>";
    private string newPath;
    //private string imageFolder;
    //private string directory;
    private string imageFileName;
    private string zipFileName;
    private string filePathName;
    private string zipPathName;
    private string plyPathName;
    private string DCIM = "/storage/emulated/0/DCIM";
    private int imageCounter = 0;
    private int compressionLevel = 9;
    private const long BUFFER_SIZE = 4096;
    #endregion

    public LevelLoader levelLoader;
    private int counter;

    // Start is called before the first frame update
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

            // Check connection to TCP Server
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

            // Create image directory
            newPath = Application.persistentDataPath + "/Screenshots/";

            // Check if snapshots path already exists
            if (!Directory.Exists(newPath))
            {
                // Create path folder
                Debug.Log("Creating screenshots directory!");
                Directory.CreateDirectory(newPath);
            }

            imageCounter = 0;

            // Take 10 pictures using the Native Camera
            StartCoroutine(TakePicture(512));

            // Compress the pictures and send them all in a .ZIP file to REST API
            string folderpathname = DCIM + newPath;
            StartCoroutine(CompressToZIP(folderpathname));
        }
    }

    private void FixedUpdate()
    {
        //Don't attempt to use the camera if it is already open
        //if (NativeCamera.IsCameraBusy())
        //{
        //    return;
        //}
        //else
        //{
        //    if (takePicture)
        //    {
        //        // Take 10 pictures with the camera and send them in a .ZIP file to REST API
        //        // If the captured image's width and/or height is greater than 512px, down-scale it
        //        TakePicture(512);
        //    }

        //}
    }

    private void PickImage(int maxSize, string imageFileName)
    {
        // Don't attempt to pick media from Gallery/Photos if ano0ther media pick operation is already in progress
        if (NativeGallery.IsMediaPickerBusy())
        {
            return;
        }
        // Pick image from gallery
        NativeGallery.Permission permission = NativeGallery.GetImageFromGallery((path) =>
        {
            Debug.Log("Image path: " + path);
            if (path != null)
            {
                // Create Texture from selected image
                Texture2D texture = NativeGallery.LoadImageAtPath(path, maxSize);
                if (texture == null)
                {
                    Debug.Log("Could not load texture from " + path);
                    return;
                }
                // Save texture in a .jpg image file
                byte[] bytes = texture.EncodeToJPG();
                string imagePathName = newPath + imageFileName;
                using (FileStream filestream = new FileStream(imagePathName, FileMode.Create, FileAccess.Write, FileShare.ReadWrite))
                {
                    filestream.Write(bytes, 0, bytes.Length);
                }
                // Wait for a sec
                Thread.Sleep(1000);
                // Check if image file is created in image folder
                FileInfo fileInfo = new FileInfo(imagePathName);
                // Get image file size;
                if (!fileInfo.Exists)
                {
                    Debug.Log("Could not locate image file!");
                    return;
                }
            }
        }, "Select a JPG image", "image/jpg");
        Debug.Log("Permission result: " + permission);
    }

    private void RemovePictures(string location)
    {
        // Remove old screenshot files from folder
        if (Directory.Exists(location))
        {
            Debug.Log("Removing old screenshots...");
            string[] filepaths = Directory.GetFiles(location, "*.jpg");
            Debug.Log("Filepaths: " + filepaths.Length);
            foreach (string filepath in filepaths)
            {
                Debug.Log("Deleting files in folder!");
                File.Delete(filepath);
            }
        }
    }

    private static void RemoveFolders(string location)
    {
        // Remove old screenshot folders
        if (Directory.Exists(location))
        {
            foreach (var directory in Directory.GetDirectories(location))
            {
                //RemoveFolders(directory);
                Debug.Log("Directory: " + directory.ToString());
                if (Directory.GetFileSystemEntries(directory).Length == 0)
                {
                    Debug.Log("Deleting file folder!");
                    Directory.Delete(directory, false);
                }
            }
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
    /// Take pictures using NativeCamera and send all pictures in a zip file via REST API
    /// </summary>
    private IEnumerator TakePicture(int maxSize)
    {
        takePicture = true;
        yield return new WaitForEndOfFrame();
        while (imageCounter < 10)
        {
            NativeCamera.Permission permission = NativeCamera.TakePicture((path) =>
            {
                Debug.Log("Image path: " + path);
                // Create zip file name
                imageFileName = imageCounter.ToString() + ".jpg";
                if (path != null)
                {
                    // Take picture and save them in native gallery
                    NativeGallery.SaveImageToGallery(path, newPath, imageFileName);
                    // Sleep for 1000ms = 1 sec
                    Thread.Sleep(1000);
                    // Get image file path name
                    filePathName = DCIM + newPath + "/" + imageFileName;
                    // Get image file information
                    FileInfo fileInfo = new FileInfo(filePathName);
                    string filesize;
                    // Get image file size;
                    if (fileInfo.Exists)
                    {
                        long size = fileInfo.Length;
                        filesize = size.ToString();
                    }
                    else
                    {
                        Debug.Log("Could not locate image file!");
                        return;
                    }
                }
            }, maxSize);
            imageCounter++;
            Debug.Log("Image Counter: " + imageCounter);
        }
        if (imageCounter == 10)
        {
            takePicture = false;
        }
    }

    /// <summary>
    /// Wait stop for CompressToZIP Coroutine for waitTime seconds
    /// </summary>
    private IEnumerator wait(float waitTime)
    {
        float counter = 0;
        while (counter < waitTime)
        {
            //Increment Timer until counter >= waitTime
            counter += Time.deltaTime;
            Debug.Log("We have waited for: " + counter + " seconds");
            if (counter == waitTime)
            {
                // Quit function
                yield break;
            }
            // Wait for a frame so that Unity does not freeze
            yield return null;
        }
    }

    /// <summary>
    /// Compress all the images files and send them in a .ZIP file via REST API
    /// </summary>
    private IEnumerator CompressToZIP(string folder)
    {
        compressPictures = true;
        float waitTime = 15;
        yield return wait(waitTime);
        if (imageCounter == 10)
        {
            Debug.Log("File Directory: " + folder);
            // Check if folder exists
            DirectoryInfo directorytInfo = new DirectoryInfo(folder);
            if (directorytInfo.Exists)
            {
                // Sleep for 1000ms = 1 sec
                Thread.Sleep(1000);
                try
                {
                    string[] files = Directory.GetFiles(folder, "*.jpg");
                    zipFileName = "FAMEST" + System.DateTime.Now.ToString("yyyy_MM_dd_HH_mm_ss") + ".zip";
                    //zipFileName = imageFolder + ".zip";
                    //zipPathName = directoryFolder + "/" + zipFileName;
                    zipPathName = Path.Combine(folder, zipFileName);
                    // Sleep for 3000ms = 3 sec
                    Thread.Sleep(3000);
                    // Create ZIP Output Stream
                    Debug.Log("Creating .ZIP Output Stream!");
                    using (ZipOutputStream OutputStream = new ZipOutputStream(File.Create(zipPathName)))
                    {
                        // Define the compression level
                        // 0 - store only to 9 - means best compression
                        OutputStream.SetLevel(9);
                        byte[] buffer = new byte[4096];
                        Debug.Log("Files Length: " + files.Length);
                        foreach (string file in files)
                        {
                            // Using GetFileName makes the result compatible with XP as the resulting path is not absolute.
                            ZipEntry entry = new ZipEntry(Path.GetFileName(file));
                            Debug.Log("ZIP Entry: " + entry.ToString());
                            // Setup the entry data as required.
                            // Crc and size are handled by the library for seakable streams so no need to do them here.
                            // Could also use the last write time or similar for the file.
                            entry.DateTime = DateTime.Now;
                            OutputStream.PutNextEntry(entry);
                            using (FileStream fs = File.OpenRead(file))
                            {
                                // Using a fixed size buffer here makes no noticeable difference for output
                                // but keeps a lid on memory usage.
                                int sourceBytes;
                                do
                                {
                                    sourceBytes = fs.Read(buffer, 0, buffer.Length);
                                    OutputStream.Write(buffer, 0, sourceBytes);
                                } while (sourceBytes > 0);
                            }
                        }
                        // Finish/Close arent needed strictly as the using statement does this automatically but it is important to ensure trailing information for a Zip file 
                        // is appended.  Without this the created file would be invalid.
                        OutputStream.Finish();
                        // Close is important to wrap things up and unlock the file.
                        OutputStream.Close();
                        Debug.Log("Files successfully compressed!");
                    }
                }
                catch (Exception ex)
                {
                    Debug.Log("Exception during processing: " + ex);
                }
                // Sleep for 1000ms = 1 sec
                Thread.Sleep(1000);
                //zipPathName = folder + "/" + zipFileName;
                //zipPathName = folder + "/" + "photos_src.zip";
                //zipPathName = DCIM + Application.persistentDataPath + "/Screenshots/photos_src.zip";
                zipPathName = DCIM + Application.persistentDataPath + "/Screenshots/" + zipFileName;
                Debug.Log("ZIP File: " + zipPathName);
                // Get zip file information
                FileInfo fileInfo = new FileInfo(folder + "/" + "photos_src.zip");
                string filesize;
                bool fileExists = false;
                // Get zip file size
                if (fileInfo.Exists)
                {
                    Debug.Log("Located file!");
                    long size = fileInfo.Length;
                    filesize = size.ToString();
                    fileExists = true;
                }
                else
                {
                    Debug.Log("Could not locate file!");
                    yield break;
                }
                if (fileExists)
                {
                    // Set server certificate validation
                    ServicePointManager.ServerCertificateValidationCallback = MyRemoteCertificateValidationCallback;
                    // Create client for REST API and send POST request with video file
                    var client = new RestClient("https://cloud1cvig.ccg.pt:19100/api/process");
                    client.Timeout = -1;
                    var request = new RestRequest(Method.POST);
                    request.AddFile("file_data", zipPathName);
                    Debug.Log("Client has sent zip file!");
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
                        // Reset image counter
                        imageCounter = 0;
                        //takePicture = false;
                        // Remove old pictures
                        RemovePictures(DCIM + Application.persistentDataPath + "/Screenshots/");
                        // Sleep for a sec
                        Thread.Sleep(1000);
                        // Remove old folder
                        //RemoveFolders(Path.Combine(DCIM, newPath));
                        // Debug
                        Debug.Log(response.StatusCode);
                        levelLoader.LoadLevel(3);
                    });
                }
            }
            else
            {
                Debug.Log("File Directory does not exist!");
            }
            compressPictures = false;
        }
    }

    /// <summary>
    /// Copies ZIP input file stream to another ZIP output file stream
    /// </summary>
    /// <param name="inputStream"></param>
    /// <param name="outputStream"></param>
    private void CopyZIPStream(FileStream inputStream, Stream outputStream)
    {
        long bufferSize = inputStream.Length < BUFFER_SIZE ? inputStream.Length : BUFFER_SIZE;
        byte[] buffer = new byte[bufferSize];
        int bytesRead = 0;
        long bytesWritten = 0;
        while ((bytesRead = inputStream.Read(buffer, 0, buffer.Length)) != 0)
        {
            outputStream.Write(buffer, 0, bytesRead);
            bytesWritten += bufferSize;
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
    /// Listen for incoming data;
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

    // Update is called once per frame
    void Update()
    {

    }
}
