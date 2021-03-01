using System.Collections;
using System.Collections.Generic;
using UnityEngine;


/// <summary>
/// Object interface so it may be touchable
/// </summary>
public interface SelectionReceiver {
    void objectSelected(GameObject obj);
}
