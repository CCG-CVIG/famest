using System;
using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Text;
using System.IO;
using UnityEngine;
using Random = UnityEngine.Random;
using System.Globalization;

[RequireComponent(typeof(ParticleSystem))]

public class ParticleScriptV2 : MonoBehaviour
{
    #region private members
    private TcpClient socketConnection;
    private Thread clientReceiveThread;
    private string separator = "<SEPARATOR>";
    private string filename;
    private int filesize;
    private string filePathName;
    private FileStream fileStream;
    private bool firstMessage = true;
    private bool displayPly = false;
    private bool getlines = false;
    #endregion

    ParticleSystem.Particle[] particles;
    ParticleSystem particleSystem;
    int numAlive;
    bool itRan;
    string[] lines;

    private void LateUpdate()
    {
        if (getlines)
        {
            InitializeIfNeeded();
            particleSystem = GetComponent<ParticleSystem>();
            ParticleSystem.EmitParams emitOverride = new ParticleSystem.EmitParams();
            particleSystem.SetParticles(particles, numAlive);
            particleSystem.Emit(emitOverride, lines.Length - 12);
            numAlive = particleSystem.GetParticles(particles);
            if (itRan == false)
            {
                CallOnce();
            }
        }

    }

    private void CallOnce()
    {


        for (int i = 11; i < particles.Length; i++)
        //for (int i = 0; i < 5000; i++)
        {
            string[] items = lines[i].Split(' ');

            //particles[i].position = new Vector3(Random.Range(-500f,500f), Random.Range(-500f, 500f), Random.Range(-500f, 500f));
            //particles[i].velocity = new Vector3(0,0,0);
           
            particles[i].position = new Vector3(float.Parse(items[1]),
                                                float.Parse(items[2]),
                                                float.Parse(items[3]));
            particles[i].velocity = new Vector3(float.Parse(items[4]),
                                                float.Parse(items[5]),
                                                float.Parse(items[6]));

        }
        itRan = true;
    }

    private void InitializeIfNeeded()
    {
        if (particleSystem == null)
            particleSystem = GetComponent<ParticleSystem>();

        if (particles == null || particles.Length < particleSystem.main.maxParticles)
            particles = new ParticleSystem.Particle[particleSystem.main.maxParticles];
        
    }

    // Start is called before the first frame update
    void Start()
    {
        filePathName = Application.persistentDataPath + "/surface_cloud.ply";
        //Debug.Log(filePathName);
        if (File.Exists(filePathName))
        {
            //Debug.Log("Point cloud file received!");
            // Draw lines from .ply file
            Thread.Sleep(5000);
            using (FileStream newfileStream = new FileStream(filePathName, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
            {
                if (newfileStream.CanRead)
                {
                    //lines = System.IO.File.ReadAllLines(filePathName);
                    List<string> filelines = new List<string>();
                    using (StreamReader reader = new StreamReader(newfileStream))
                    {
                        while (!reader.EndOfStream)
                        {
                            filelines.Add(reader.ReadLine());
                        }
                    }
                    // Convert List<string> into string[]
                    lines = filelines.ToArray();
                    if (lines != null)
                    {
                        /*
                        foreach (string line in lines)
                        {
                            Debug.Log("Line: " + line);
                        }
                        */
                        getlines = true;
                    }
                }
            }
        }
        else
        {
            Debug.Log("Point cloud file not found!");
        }
    }

    private IEnumerable<string> ReadLines(Func<Stream> streamProvider, Encoding encoding)
    {
        using (var stream = streamProvider())
        using (var reader = new StreamReader(stream, encoding))
        {
            string line;
            while ((line = reader.ReadLine()) != null)
            {
                yield return line;
            }
        }
    }

    /// <summary>
    /// Setup TCP socket connection
    /// </summary>
    //private void ConnectToTCPServer()
    //{
    //    try
    //    {
    //        clientReceiveThread = new Thread(new ThreadStart(ListenForData));
    //        clientReceiveThread.IsBackground = true;
    //        clientReceiveThread.Start();
    //    }
    //    catch(Exception e)
    //    {
    //        Debug.Log("On client connection exception " + e);
    //    }
    //}

    /// <summary>
    /// Runs in background of clientReceiveThread;
    /// Listens for incoming data;
    /// </summary>
    //private void ListenForData()
    //{
    //    try
    //    {
    //        socketConnection = new TcpClient("192.168.1.171", 30000);
    //        Byte[] messageBytes = new byte[1024];
    //        while (true) {
    //            // Get stream object for reading
    //            using (NetworkStream stream = socketConnection.GetStream())
    //            {
    //                StreamReader reader = new StreamReader(stream);
    //                int length;
    //                // Read incoming message data stream into byte array
    //                while ((length = stream.Read(messageBytes, 0, messageBytes.Length)) != 0)
    //                {
    //                    var incomingMessageData = new byte[length];
    //                    Array.Copy(messageBytes, 0, incomingMessageData, 0, length);
    //                    // Convert byte stream to string
    //                    string serverMessage = Encoding.ASCII.GetString(incomingMessageData);
    //                    Debug.Log("Server message received as: " + serverMessage);
    //                    // Split file information according to separator
    //                    filename = Path.GetFileName(serverMessage.Substring(0, serverMessage.IndexOf(separator)));
    //                    //string filesize = serverMessage.Substring(1, serverMessage.IndexOf(separator));
    //                    Debug.Log("Filename: " + filename);
    //                    // Get full path name
    //                    //filePathName = Application.persistentDataPath + "/" + filename;
    //                }
    //            }
    //        }
    //    }
    //    catch(SocketException socketException)
    //    {
    //        Debug.Log("Socket exception: " + socketException);
    //    }
    //}

    // Update is called once per frame
    void Update()
    {
        
    }
}
