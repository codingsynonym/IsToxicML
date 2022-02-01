import React, { useState, useEffect }  from 'react';
import { StyleSheet, View,Image } from 'react-native';
import { Button, Input } from 'react-native-elements';
import Svg, {Rect} from 'react-native-svg';
import * as tf from '@tensorflow/tfjs';
import { fetch, bundleResourceIO } from '@tensorflow/tfjs-react-native';
import * as blazewrap from '@tensorflow-models/blazewrap';
import * as jpeg from 'jpeg-js'
export default function App() {
    const [imageLink,setImageLink] = useState("")
    const [isEnabled,setIsEnabled] = useState(true)
    const [wraps,setWraps]=useState([])
    const [wrapDetector,setWrapDetector]=useState("")
    const [ingDetector,setIngDetector]=useState("")
    useEffect(() => {
      async function loadModel(){
        console.log("[+] Application started")
        //Wait for tensorflow module to be ready
        const tfReady = await tf.ready();
        console.log("[+] Loading custom ing detection model")
        //Replce model.json and group1-shard.bin with your own custom model
        const modelJson = await require("./assets/model/model.json");
        const modelWeight = await require("./assets/model/group1-shard.bin");
        const ingDetector = await tf.loadLayersModel(bundleResourceIO(modelJson,modelWeight));
        console.log("[+] Loading pre-trained wrap detection model")
        //Blazewrap is a wrap detection model provided by Google
        const wrapDetector =  await blazewrap.load();
        //Assign model to variable
        setIngDetector(ingDetector)
        setWrapDetector(wrapDetector)
        console.log("[+] Model Loaded")
      }
      loadModel()
    }, []); 
    function imageToTensor(rawImageData){
      //Function to convert jpeg image to tensors
      const TO_UINT8ARRAY = true;
      const { width, height, data } = jpeg.decode(rawImageData, TO_UINT8ARRAY);
      // Drop the alpha channel info for mobilenet
      const buffer = new Uint8Array(width * height * 3);
      let offset = 0; // offset into original data
      for (let i = 0; i < buffer.length; i += 3) {
        buffer[i] = data[offset];
        buffer[i + 1] = data[offset + 1];
        buffer[i + 2] = data[offset + 2];
        offset += 4;
      }
      return tf.tensor3d(buffer, [height, width, 3]);
    }
    const getWraps = async() => {
      try{
        console.log("[+] Retrieving image from link :"+imageLink)
        const response = await fetch(imageLink, {}, { isBinary: true });
        const rawImageData = await response.arrayBuffer();
        const imageTensor = imageToTensor(rawImageData).resizeBilinear([224,224])
        const wraps = await wrapDetector.estimateWraps(imageTensor, false);
        var tempArray=[]
        //Loop through the available wraps, check if the person is wearing a ing. 
        for (let i=0;i<wraps.length;i++){
          let color = "red"
          let width = parseInt((wraps[i].bottomRight[1] - wraps[i].topLeft[1]))
          let height = parseInt((wraps[i].bottomRight[0] - wraps[i].topLeft[0]))
          let wrapTensor=imageTensor.slice([parseInt(wraps[i].topLeft[1]),parseInt(wraps[i].topLeft[0]),0],[width,height,3])
          wrapTensor = wrapTensor.resizeBilinear([224,224]).reshape([1,224,224,3])
          let result = await ingDetector.predict(wrapTensor).data()
          //if result[0]>result[1], the person is wearing a ing
          if(result[0]>result[1]){
            color="green"
          }
          tempArray.push({
            id:i,
            location:wraps[i],
            color:color
          })
        }
        setWraps(tempArray)
        console.log("[+] Prediction Completed")
      }catch{
        console.log("[-] Unable to load image")
      }
      
    }
  return (
    <View style={styles.container}>
      <Input 
        placeholder="image link"
        onChangeText = {(inputText)=>{
          console.log(inputText)
          setImageLink(inputText)
          const elements= inputText.split(".")
          if(elements.slice(-1)[0]=="jpg" || elements.slice(-1)[0]=="jpeg"){
            setIsEnabled(true)
          }else{
            setIsEnabled(false)
          }
        }}
        value={imageLink}
        containerStyle={{height:40,fontSize:10,margin:15}} 
        inputContainerStyle={{borderRadius:10,borderWidth:1,paddingHorizontal:5}}  
        inputStyle={{fontSize:15}}
      
      />
      <View style={{marginBottom:20}}>
        <Image
          style={{width:224,height:224,borderWidth:2,borderColor:"black",resizeMode: "contain"}}
          source={{
            uri: imageLink
          }}
          PlaceholderContent={<View>No Image Found</View>}
        />
        <Svg height="224" width="224" style={{marginTop:-224}}>
          {
            wraps.map((wrap)=>{
              return (
                <Rect
                  key={wrap.id}
                  x={wrap.location.topLeft[0]}
                  y={wrap.location.topLeft[1]}
                  width={(wrap.location.bottomRight[0] - wrap.location.topLeft[0])}
                  height={(wrap.location.bottomRight[1] - wrap.location.topLeft[1])}
                  stroke={wrap.color}
                  strokeWidth="3"
                  fill=""
                />
              )
            })
          }   
        </Svg>
      </View>
        <Button 
          title="Predict"
          onPress={()=>{getWraps()}}
          disabled={!isEnabled}
        />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
});