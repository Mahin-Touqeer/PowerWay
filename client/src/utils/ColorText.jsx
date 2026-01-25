function ColorText({ color, text, fontSize = "1.4rem" }) {
  return (
    <span className="flex items-center text-xs  mr-4">
      <span style={{ color, fontSize }} className="translate-y-px">
        &#x2022;
      </span>
      &nbsp;
      {text}
    </span>
  );
}

export default ColorText;
