<template>
  <div id="Footer">
    <p>{{ msg }}</p>
  </div>
</template>
<script>
export default {
  name: "Footer",
  data() {
    return {
      msg: "Copyright 2024",
    };
  },
};
</script>
<style scoped>
#Footer {
  /*background:#F2F6FC;*/
  padding: 6px;
  border-radius: 5px;
  width: 80%;
  height: 80px;
  margin: 20px auto;
  margin-top: 200px;
}

p {
  color: #2e87cf;
  text-align: center;
  margin: 30px auto;
  font-size: 1.1em;
}
</style>


