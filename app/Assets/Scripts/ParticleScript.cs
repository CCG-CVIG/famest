using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using Random = UnityEngine.Random;
using System.Globalization;

[RequireComponent(typeof(ParticleSystem))]

public class ParticleScript : MonoBehaviour
{

    ParticleSystem.Particle[] particles;
    private new ParticleSystem particleSystem;
    int numAlive;
    bool itRan;
    string[] lines;
    public float width, length = 0;

    private void LateUpdate()
    {
        InitializeIfNeeded();
        particleSystem = GetComponent<ParticleSystem>();
        ParticleSystem.EmitParams emitOverride = new ParticleSystem.EmitParams();
        particleSystem.SetParticles(particles,numAlive);
        particleSystem.Emit(emitOverride, lines.Length-12);
        numAlive = particleSystem.GetParticles(particles);
        if (itRan == false)
        {
            CallOnce();
        }

    }

    private void CallOnce()
    {
        for (int i = 11; i < particles.Length; i++)
        {
            string[] items = lines[i].Split(' ');

            
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
        var path = Application.persistentDataPath + "/surface_cloud.ply";
        Debug.Log(path);
        lines = System.IO.File.ReadAllLines(path);
    }

    
}
