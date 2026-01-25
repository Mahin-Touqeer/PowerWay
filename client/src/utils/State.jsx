function State({ status }) {
  return (
    <span
      className="px-3 rounded-md"
      style={{
        color: `var(--${status})`,
        backgroundColor: `var(--${status}-background)`,
      }}
    >
      {status}
    </span>
  );
}

export default State;
