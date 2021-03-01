using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

/// <summary>
/// Listens for touch events and performs an AR raycast from the screen touch point.
/// AR raycasts will only hit detected trackables like feature points and planes.
///
/// If a raycast hits a trackable, the <see cref="placedPrefab"/> is instantiated
/// and moved to the hit position.
/// </summary>
[RequireComponent(typeof(ARRaycastManager))]
public class PlaceOnPlane : MonoBehaviour
{
    [SerializeField]
    [Tooltip("Instantiates this prefab on a plane at the touch location.")]
    GameObject m_PlacedPrefab;

    /// <summary>
    /// The prefab to instantiate on touch.
    /// </summary>
    public GameObject placedPrefab
    {
        get { return m_PlacedPrefab; }
        set { m_PlacedPrefab = value; }
    }

    /// <summary>
    /// The prefab to instantiate on touch.
    /// </summary>
    public GameObject[] placedPrefab2;

    /// <summary>
    /// The object instantiated as a result of a successful raycast intersection with a plane.
    /// </summary>
    public GameObject particleHolder { get; private set; }

    /// <summary>
    /// The object instantiated as a result of a successful raycast intersection with a plane.
    /// </summary>
    public GameObject shoeHolder { get; private set; }

    private int index = 0;
    private float length, width = 21.0f;
    private string filePathName;
    private float height = 21.0f;

    void Awake()
    {
        m_RaycastManager = GetComponent<ARRaycastManager>();
        getFootSize();
    }

    bool TryGetTouchPosition(out Vector2 touchPosition)
    {
#if UNITY_EDITOR
        if (Input.GetMouseButton(0))
        {
            var mousePosition = Input.mousePosition;
            touchPosition = new Vector2(mousePosition.x, mousePosition.y);
            return true;
        }
#else
        if (Input.touchCount > 0)
        {
            touchPosition = Input.GetTouch(0).position;
            return true;
        }
#endif

        touchPosition = default;
        return false;
    }

    void Update()
    {
        if (!TryGetTouchPosition(out Vector2 touchPosition))
            return;

        if (m_RaycastManager.Raycast(touchPosition, s_Hits, TrackableType.PlaneWithinPolygon))
        {
            // Raycast hits are sorted by distance, so the first one
            // will be the closest hit.
            var hitPose = s_Hits[0].pose;

            if (particleHolder == null)
            {
                //particleHolder = Instantiate(m_PlacedPrefab, hitPose.position, hitPose.rotation);
                particleHolder = Instantiate(m_PlacedPrefab, hitPose.position, hitPose.rotation);
                particleHolder.transform.Rotate(-90.00f,90.00f,0, Space.Self);
                shoeHolder = Instantiate(placedPrefab2[this.index], hitPose.position, hitPose.rotation);
                shoeHolder.transform.Rotate(0, 5, 0, Space.Self);
                float scale = this.length / 21.0f;
                shoeHolder.transform.localScale = new Vector3(scale, scale, scale);
            }
            else
            {
                particleHolder.transform.position = hitPose.position;
                shoeHolder.transform.position = hitPose.position;
            }
        }
    }

    public void onClick(int newIndex)
    {

        if (shoeHolder != null)
        {
            GameObject.Destroy(shoeHolder);
        }
        //transform.localPosition = new Vector3(0.0f, 0.0f, 0.0f);
        this.index = newIndex;
        shoeHolder = Instantiate(placedPrefab2[this.index]);
        shoeHolder.transform.position = particleHolder.transform.position;

        shoeHolder.transform.Rotate(0, 5, 0, Space.Self);
        float scale = this.length / 21.0f;
        shoeHolder.transform.localScale = new Vector3(scale, scale, scale);
    }

    private void getFootSize()
    {
        filePathName = Application.persistentDataPath + "/surface_cloud.ply";
        if (File.Exists(filePathName))
        {
            string[] lines = System.IO.File.ReadAllLines(filePathName);
            float coordx_min = 0.0f, coordx_max = 0.0f;
            float coordy_min = 0.0f, coordy_max = 0.0f;
            float coordz_min = 0.0f, coordz_max = 0.0f;
            for (int i = 0; i < lines.Length - 11; i++)
            {
                string[] items = lines[i + 11].Split(' ');
                if (coordx_min > float.Parse(items[1])) { coordx_min = float.Parse(items[1]); }
                if (coordx_max < float.Parse(items[1])) { coordx_max = float.Parse(items[1]); }
                if (coordy_min > float.Parse(items[2])) { coordy_min = float.Parse(items[2]); }
                if (coordy_max < float.Parse(items[2])) { coordy_max = float.Parse(items[2]); }
                if (coordz_min > float.Parse(items[3])) { coordz_min = float.Parse(items[3]); }
                if (coordz_max < float.Parse(items[3])) { coordz_max = float.Parse(items[3]); }
            }
            this.length = coordx_max - coordx_min;
            this.width = coordy_max - coordy_min;
            this.height = coordz_max - coordz_min;
        }
    }

    static List<ARRaycastHit> s_Hits = new List<ARRaycastHit>();

    ARRaycastManager m_RaycastManager;
}
