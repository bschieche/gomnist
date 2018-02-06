package controller

const pageTemplate = `<!DOCTYPE HTML>
<html>
  <head>
    <title>MNIST</title>
    <style>.error{color:#FF0000;}</style>
    <meta name="author" content="Bettina Schieche" />
    {{ if .ID }}<meta http-equiv="refresh" content="5">{{ end }}
  </head>
<body>
  <h2>Learning handwritten digits</h2>
  <h3>Use a neural network with one hidden layer for handwritten digit recognition.</h3>
  <p>Please enter parameters below (comma or space-seprated). Avoid duplicates to not waste computational cost!</p>
  <form action="/submit" method="POST">
   <table>
    <tr>
     <td> <center><b>model parameter names</b></center></td>
     <td> <center><b>list of inputs</b></center></td>
     <td> <center><b>Best loss obtained for</b></center></td>
    </tr>
    <tr>
     <td> </td>
     <td> </td>
     <td> </td>
    </tr>
    <tr>
     <td> lambda (weight decay):</td>
     <td> <input type="text" name="lambda" size="30"></td>
     <td> {{ .Lambda }}</td>
    </tr>
    <tr>
     <td> number of hidden units:</td>
     <td> <input type="text" name="numHidden" size="30"></td>
     <td> {{ .NumHidden }}</td>
    </tr>
    <tr>
     <td> number of iterations:</td>
     <td> <input type="text" name="numIters" size="30"></td>
     <td> {{ .NumIters }}</td>
    </tr>
    <tr>
     <td> learning rate (between 0 and 1):</td>
     <td> <input type="text" name="learningRate" size="30"></td>
     <td> {{ .LearningRate }}</td>
    </tr>
    <tr>
     <td> momentum multiplier:</td>
     <td> <input type="text" name="momentum" size="30"></td>
     <td> {{ .Momentum }}</td>
    </tr>
    <tr>
     <td> mini batch size:</td>
     <td> <input type="text" name="batchSize" size="30"></td>
     <td> {{ .BatchSize }}</td>
    </tr>
   </table>
    <input type="submit" value="Compute">
  </form>
  <p>Best classification loss: {{ .Loss }}<br>
  Best error rate: {{ .ErrorRate }}</p>
  <hr />
  <table border="1">
   <tr>
    <td><center>Learning rate</center></td>
    <td><center>Randomly chosen misclassified image</center></td>
   </tr>
   <tr>
    <td><center><img src="/render?id={{ .ID }}&typ=learningCurve" /></center></td>
    <td><center><img src="/render?id={{ .ID }}&typ=digit" /></center></td>
   </tr>
  </table>
</body>
</html>`
