function KeyValue({ prop, value }) {
  return (
    <>
      <div className="text-gray-500">{prop}</div>
      <div className="mb-3 font-semibold">{value}</div>
    </>
  );
}

export default KeyValue;
