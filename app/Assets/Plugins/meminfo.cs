
using UnityEngine;
using System;
using System.Collections;

#if UNITY_ANDROID
	using System.Text;
	using System.Text.RegularExpressions;
	using System.IO;
#endif

#if UNITY_IPHONE || UNITY_IOS
	using System.Runtime.InteropServices;
#endif

// On Android Free Ram = memfree + cached

// On iOS to get the available memory use (free + inactive) !

public class meminfo  {
#if !UNITY_EDITOR && !UNITY_WEBPLAYER

	#if UNITY_ANDROID
		public struct meminf{
			//all numbers are in kiloBytes
			public int memtotal;
			public int memfree;
			public int memavailable;
			public int active;
			public int inactive;
			public int cached;
			public int swapcached;
			public int swaptotal;
			public int swapfree;
		}
		
		public static meminf minf = new meminf();
		
		private static Regex re = new Regex(@"\d+");
		
		public static bool getMemInfo(){
			
			if(!File.Exists("/proc/meminfo")) return false;
		
			FileStream fs = new FileStream("/proc/meminfo", FileMode.Open, FileAccess.Read, FileShare.Read);
			StreamReader sr = new StreamReader(fs);
			
			string line;
			while((line = sr.ReadLine())!=null){
				line = line.ToLower().Replace(" ","");
				if(line.Contains("memtotal")){ minf.memtotal = mVal(line); }
				if(line.Contains("memfree")){ minf.memfree = mVal(line); }
				if(line.Contains("memavailable")){ minf.memavailable = mVal(line); }
				if(line.Contains("active")){ minf.active = mVal(line); }
				if(line.Contains("inactive")){ minf.inactive = mVal(line); }
				if(line.Contains("cached") && !line.Contains("swapcached")){ minf.cached = mVal(line); }
				if(line.Contains("swapcached")){ minf.swapcached = mVal(line); }
				if(line.Contains("swaptotal")){ minf.swaptotal = mVal(line); }
				if(line.Contains("swapfree")){ minf.swapfree = mVal(line); }
			}
			
			sr.Close(); fs.Close(); fs.Dispose();
			return true;
		}
		
		private static int mVal(string s){
			Match m = re.Match(s); return int.Parse(m.Value);
		}
	
		public static void gc_Collect() {
			var jc = new AndroidJavaClass("java.lang.System");
			jc.CallStatic("gc");
			jc.Dispose();
		}

	#endif
	
	#if UNITY_IPHONE || UNITY_IOS
	
		public struct meminf{
			//all numbers are in bytes
			public long memtotal;
			public long memfree;
			public long memused;

			public long memactive;
			public long meminactive;
			public long memwired;
		}
	
		public static meminf minf = new meminf();
		
		[DllImport("__Internal")]
		private static extern long igetRam(int what);
	
	

		//to get the available memory on ios use (free + inactive) !

		public static bool getMemInfo(){
		
			long rt;

			minf.memtotal = igetRam(0); // physical memory

			rt = minf.memfree = igetRam(1);//free
			rt = minf.memused = igetRam(2);//used

			rt = minf.memactive = igetRam(3);//active
			rt = minf.meminactive = igetRam(4);//inactive
			rt = minf.memwired= igetRam(5);//wired


			if(rt==-1) return false;
			
			return true;
			
		}
	
	#endif

#endif
}

