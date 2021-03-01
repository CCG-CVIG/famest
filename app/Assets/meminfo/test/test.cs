using UnityEngine;
using System.Collections;

public class test : MonoBehaviour {
#if !UNITY_WEBPLAYER
	void Start () {
		#if (UNITY_ANDROID || UNITY_IPHONE || UNITY_IOS) && !UNITY_EDITOR
			//call this every time you want to get updated data
			//if it returns false no memory data was retrieved
			bool mi = meminfo.getMemInfo();
			if(!mi) Debug.Log("Could not get Memory Info");
		#endif

	}

	
	void Update () {
		if (Input.GetKeyDown(KeyCode.Escape)){Application.Quit();}
	}
	
	void OnGUI(){

		#if (UNITY_ANDROID || UNITY_IPHONE || UNITY_IOS) && !UNITY_EDITOR
			if(GUI.Button(new Rect(10,Screen.height-50,180,40),"Get MemInfo")){meminfo.getMemInfo();}
			#if UNITY_ANDROID 
				if(GUI.Button(new Rect(200,Screen.height-50,180,40),"native Gc Collect")){meminfo.gc_Collect();}
			#endif
		#endif

	#if !UNITY_EDITOR
		#if UNITY_ANDROID
			GUI.Label(new Rect(50,10,250,40),"memtotal: " + meminfo.minf.memtotal.ToString() + " kb");
			GUI.Label(new Rect(50,50,250,40),"memfree: " + meminfo.minf.memfree.ToString() + " kb");
			GUI.Label(new Rect(50,90,250,40),"memavailable: " + meminfo.minf.memavailable.ToString() + " kb");
			GUI.Label(new Rect(50,130,250,40),"active: " + meminfo.minf.active.ToString() + " kb");
			GUI.Label(new Rect(50,170,250,40),"inactive: " + meminfo.minf.inactive.ToString() + " kb");
			GUI.Label(new Rect(50,210,250,40),"cached: " + meminfo.minf.cached.ToString() + " kb");
			GUI.Label(new Rect(50,250,250,40),"swapcached: " + meminfo.minf.swapcached.ToString() + " kb");
			GUI.Label(new Rect(50,290,250,40),"swaptotal: " + meminfo.minf.swaptotal.ToString() + " kb");
			GUI.Label(new Rect(50,330,250,40),"swapfree: " + meminfo.minf.swapfree.ToString() + " kb");
		#endif
		
		#if UNITY_IPHONE || UNITY_IOS
			GUI.Label(new Rect(50,10,250,40),"total: " + meminfo.minf.memtotal.ToString() + " bytes");
			GUI.Label(new Rect(50,50,250,40),"free: " + meminfo.minf.memfree.ToString() + " bytes");
			GUI.Label(new Rect(50,90,250,40),"used: " + meminfo.minf.memused.ToString() + " bytes");	
				
			GUI.Label(new Rect(50,130,250,40),"active: " + meminfo.minf.memactive.ToString() + " bytes");
			GUI.Label(new Rect(50,170,250,40),"inactive: " + meminfo.minf.meminactive.ToString() + " bytes");
			GUI.Label(new Rect(50,210,250,40),"wired: " + meminfo.minf.memwired.ToString() + " bytes");	
		#endif
	#endif

	}
#else
	void Start () {
		Debug.Log("Does not work on WebPlayer");
	}
#endif	
}
