# Benchmark of CNNs on NVIDIA jetson

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;}
.tg .tg-baqh{text-align:center;vertical-align:top}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
.tg .tg-0lax{text-align:left;vertical-align:top}
</style>
<table class="tg">
  <tr>
    <th class="tg-c3ow" rowspan="2">Arch<br>[input size]<br>vs <br>GPU</th>
    <th class="tg-c3ow" colspan="2">MobilenetsV2 - ASPP</th>
    <th class="tg-c3ow" colspan="2">ERFNet</th>
  </tr>
  <tr>
    <td class="tg-baqh">[640x480]</td>
    <td class="tg-baqh">[320x240]</td>
    <td class="tg-baqh">[640x480]</td>
    <td class="tg-baqh">[320x240]</td>
  </tr>
  <tr>
    <td class="tg-0pky">Nano FP16</td>
    <td class="tg-0pky">2.78</td>
    <td class="tg-0pky">11.5</td>
    <td class="tg-0pky">5.5</td>
    <td class="tg-0pky">20.8</td>
  </tr>
  <tr>
    <td class="tg-0pky">TX2-fp16</td>
    <td class="tg-0pky">7</td>
    <td class="tg-0pky">28.57</td>
    <td class="tg-0pky">14.49</td>
    <td class="tg-0pky">52.6</td>
  </tr>
  <tr>
    <td class="tg-0pky">Xavier-fp16</td>
    <td class="tg-0pky">44.5</td>
    <td class="tg-0pky">142.8</td>
    <td class="tg-0pky">62.5</td>
    <td class="tg-0pky">166.6</td>
  </tr>
  <tr>
    <td class="tg-0pky">Xavier-int8</td>
    <td class="tg-0pky">62.5</td>
    <td class="tg-0pky">166.6</td>
    <td class="tg-0pky">83.5</td>
    <td class="tg-0pky">200</td>
  </tr>
  <tr>
    <td class="tg-0lax">Xavier-fp16-DLA*</td>
    <td class="tg-0lax">-</td>
    <td class="tg-0lax">-</td>
    <td class="tg-0lax">21.7</td>
    <td class="tg-0lax">45.5</td>
  </tr>
</table>